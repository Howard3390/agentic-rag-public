"""Read-only LightRAG local retrieval MCP server for the LightRAG environment."""

from __future__ import annotations

import asyncio
import base64
import json
import mmap
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv
from mcp.server import MCPServer


_retriever: tuple[Any, Any, Any, Any, Any, dict[str, Any]] | None = None
_query_param_type: Any = None
_kg_query: Any = None
_init_lock = asyncio.Lock()


def _stream_load_large_vector_index(file_name: str, original_loader: Any) -> Any:
    """Read NanoVectorDB's large JSON without duplicating it in one json.load call."""
    path = Path(file_name)
    if not path.exists() or path.stat().st_size < 25_000_000:
        return original_loader(file_name)

    import numpy as np

    with path.open("rb") as stream:
        if stream.readline().strip() != b"{":
            raise ValueError(f"Invalid NanoVectorDB index: {path}")
        dimension_line = stream.readline().strip().rstrip(b",")
        dimension = json.loads(b"{" + dimension_line + b"}")["embedding_dim"]
        if stream.readline().strip() != b'"data": [':
            raise ValueError(f"Unexpected NanoVectorDB data layout: {path}")
        records = []
        while True:
            line = stream.readline()
            if line.lstrip().startswith(b"]"):
                break
            if line.strip() != b"{":
                raise ValueError(f"Unexpected NanoVectorDB record layout: {path}")
            record_lines = [line]
            while True:
                line = stream.readline()
                if not line:
                    raise ValueError(f"Truncated NanoVectorDB index: {path}")
                record_lines.append(line)
                if line.startswith(b"    }"):
                    break
            records.append(json.loads(b"".join(record_lines).rstrip().rstrip(b",")))

        if stream.read(len(b'  "matrix": "')) != b'  "matrix": "':
            raise ValueError(f"Unexpected NanoVectorDB matrix layout: {path}")
        matrix_start = stream.tell()
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            matrix_end = mapped.rfind(b'"')
            if matrix_end <= matrix_start:
                raise ValueError(f"Invalid NanoVectorDB matrix: {path}")
            encoded = memoryview(mapped)[matrix_start:matrix_end]
            try:
                decoded = base64.b64decode(encoded)
            finally:
                encoded.release()
        matrix = np.frombuffer(decoded, dtype=np.float32).reshape(-1, dimension)
        if len(matrix) != len(records):
            raise ValueError(f"NanoVectorDB matrix and records disagree: {path}")
        return {"embedding_dim": dimension, "data": records, "matrix": matrix}


@asynccontextmanager
async def lifespan(_server: MCPServer):
    try:
        yield {}
    finally:
        if _retriever is not None:
            for storage in _retriever[:4]:
                await storage.finalize()


mcp = MCPServer("lightrag-local", lifespan=lifespan)


async def get_retriever() -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    global _retriever, _query_param_type, _kg_query
    if _retriever is not None:
        return _retriever
    async with _init_lock:
        if _retriever is not None:
            return _retriever

        source_dir = Path(os.environ.get("LIGHTRAG_SOURCE_DIR", ""))
        index_dir = Path(os.environ.get("LIGHTRAG_INDEX_DIR", ""))
        if not (source_dir / "lightrag" / "__init__.py").is_file():
            raise ValueError("LIGHTRAG_SOURCE_DIR must point to the LightRAG checkout")
        required = (
            "vdb_entities.json",
            "vdb_chunks.json",
            "graph_chunk_entity_relation.graphml",
            "kv_store_text_chunks.json",
        )
        if not index_dir.is_dir() or any(not (index_dir / name).is_file() for name in required):
            raise ValueError("LIGHTRAG_INDEX_DIR lacks the local-mode entity, graph, or chunk index")

        embedding_env = os.environ.get("LIGHTRAG_ENV_FILE")
        if not embedding_env or not Path(embedding_env).is_file():
            raise ValueError("LIGHTRAG_ENV_FILE must point to embedding configuration")
        load_dotenv(embedding_env, override=False)
        embed_key = os.environ.get("EMBEDDING_BINDING_API_KEY")
        embed_url = os.environ.get("EMBEDDING_BINDING_HOST")
        if not embed_key or not embed_url:
            raise ValueError("Embedding API settings are missing from LIGHTRAG_ENV_FILE")

        llm_env = os.environ.get("AGENTIC_RAG_LLM_ENV_FILE")
        llm_config = dotenv_values(llm_env) if llm_env and Path(llm_env).is_file() else {}
        llm_key = os.environ.get("AGENTIC_RAG_LLM_API_KEY") or llm_config.get("LLM_API_KEY")
        llm_url = os.environ.get("AGENTIC_RAG_LLM_BASE_URL") or llm_config.get("LLM_BASE_URL")
        llm_model = os.environ.get("AGENTIC_RAG_LLM_MODEL") or llm_config.get("LLM_MODEL")
        if not llm_key or not llm_url or not llm_model:
            raise ValueError("Project LLM_* settings are required for local keyword extraction")

        sys.path.insert(0, str(source_dir.resolve()))
        from lightrag import QueryParam
        from lightrag.kg.json_kv_impl import JsonKVStorage
        from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
        from lightrag.kg.networkx_impl import NetworkXStorage
        from lightrag.kg.shared_storage import initialize_share_data
        from lightrag.llm.openai import openai_embed
        from lightrag.namespace import NameSpace
        from lightrag.operate import kg_query
        from lightrag.utils import EmbeddingFunc, TiktokenTokenizer
        from openai import AsyncOpenAI
        import nano_vectordb.dbs as nano_dbs
        from functools import partial

        nano_dbs.load_storage = partial(
            _stream_load_large_vector_index, original_loader=nano_dbs.load_storage
        )

        client = AsyncOpenAI(api_key=llm_key, base_url=llm_url, timeout=60, max_retries=1)

        async def extract_keywords(prompt: str, **_kwargs: Any) -> str:
            response = await client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
                extra_body={"enable_thinking": False},
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Keyword extraction returned no text")
            return content

        async def embed_texts(texts: list[str], **kwargs: Any) -> Any:
            kwargs.pop("_priority", None)
            return await openai_embed.func(
                texts,
                model=os.environ.get("EMBEDDING_MODEL", "text-embedding-v4"),
                base_url=embed_url,
                api_key=embed_key,
                max_token_size=int(os.environ.get("EMBEDDING_TOKEN_LIMIT", "8192")),
                **kwargs,
            )

        embedding = EmbeddingFunc(
            embedding_dim=int(os.environ.get("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.environ.get("EMBEDDING_TOKEN_LIMIT", "8192")),
            func=embed_texts,
        )
        config = {
            "working_dir": str(index_dir.resolve()),
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": float(os.environ.get("COSINE_THRESHOLD", "0.2")),
            },
            "tokenizer": TiktokenTokenizer(),
            "llm_model_func": extract_keywords,
            "addon_params": {"language": "English"},
            "enable_llm_cache": False,
            "kg_chunk_pick_method": os.environ.get("KG_CHUNK_PICK_METHOD", "VECTOR"),
            "related_chunk_number": int(os.environ.get("RELATED_CHUNK_NUMBER", "5")),
        }
        initialize_share_data()
        entities = NanoVectorDBStorage(
            namespace=NameSpace.VECTOR_STORE_ENTITIES,
            workspace="",
            global_config=config,
            embedding_func=embedding,
            meta_fields={"entity_name", "source_id", "content", "file_path"},
        )
        graph = NetworkXStorage(
            namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
            workspace="",
            global_config=config,
            embedding_func=embedding,
        )
        chunks = JsonKVStorage(
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            workspace="",
            global_config=config,
            embedding_func=embedding,
        )
        chunks_vdb = NanoVectorDBStorage(
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            workspace="",
            global_config=config,
            embedding_func=embedding,
            meta_fields={"full_doc_id", "content", "file_path"},
        )
        for storage in (entities, graph, chunks, chunks_vdb):
            await storage.initialize()

        _retriever = (entities, graph, chunks, chunks_vdb, client, config)
        _query_param_type = QueryParam
        _kg_query = kg_query
        return _retriever


class _NoCache:
    global_config = {"enable_llm_cache": False}


@mcp.tool()
async def search_local(
    query: str, top_k: int = 40, chunk_top_k: int = 10, max_total_tokens: int = 30000
) -> dict[str, Any]:
    """Retrieve source-grounded chunks through LightRAG local graph search."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if min(top_k, chunk_top_k, max_total_tokens) < 1:
        raise ValueError("top_k, chunk_top_k, and max_total_tokens must be positive")
    entities, graph, chunks, chunks_vdb, _client, config = await get_retriever()
    param = _query_param_type(
        mode="local",
        top_k=top_k,
        chunk_top_k=chunk_top_k,
        max_total_tokens=max_total_tokens,
        enable_rerank=False,
        only_need_context=True,
    )
    result = await _kg_query(
        query.strip(), graph, entities, None, chunks, param, config,
        hashing_kv=_NoCache(), chunks_vdb=chunks_vdb,
    )
    payload = result.raw_data if result is not None else None
    if payload is None:
        return {"status": "success", "data": {"entities": [], "relationships": [], "chunks": []}, "metadata": {"query_mode": "local"}}
    if not isinstance(payload, dict):
        raise RuntimeError("LightRAG returned an invalid local-mode response")
    return payload


if __name__ == "__main__":
    mcp.run(transport="stdio")
