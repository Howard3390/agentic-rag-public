"""Standalone MCP server running inside the LightRAG Conda environment.

Launch this file with the LightRAG environment's Python interpreter. It imports
the specified LightRAG checkout and exposes data-only dense chunk retrieval.
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mcp.server import MCPServer


_chunks_vdb: Any = None
_query_param_type: Any = None
_naive_query: Any = None
_global_config: dict[str, Any] | None = None
_init_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_server: MCPServer):
    try:
        yield {}
    finally:
        if _chunks_vdb is not None:
            await _chunks_vdb.finalize()


mcp = MCPServer("lightrag-dense", lifespan=lifespan)


async def get_retriever() -> tuple[Any, Any, dict[str, Any]]:
    global _chunks_vdb, _query_param_type, _naive_query, _global_config
    if _chunks_vdb is not None:
        return _chunks_vdb, _naive_query, _global_config

    async with _init_lock:
        if _chunks_vdb is not None:
            return _chunks_vdb, _naive_query, _global_config

        source_dir = Path(os.environ.get("LIGHTRAG_SOURCE_DIR", ""))
        index_dir = Path(os.environ.get("LIGHTRAG_INDEX_DIR", ""))
        if not (source_dir / "lightrag" / "__init__.py").is_file():
            raise ValueError("LIGHTRAG_SOURCE_DIR must point to the LightRAG checkout")
        if not index_dir.is_dir():
            raise ValueError("LIGHTRAG_INDEX_DIR must point to an existing index")

        env_file = os.environ.get("LIGHTRAG_ENV_FILE")
        if env_file:
            if not Path(env_file).is_file():
                raise ValueError("LIGHTRAG_ENV_FILE does not exist")
            load_dotenv(env_file, override=False)

        api_key = os.environ.get("EMBEDDING_BINDING_API_KEY", "")
        base_url = os.environ.get("EMBEDDING_BINDING_HOST", "")
        if not api_key or not base_url:
            raise ValueError("Embedding API settings are missing from LIGHTRAG_ENV_FILE")

        # The installed editable package may refer to another checkout.
        sys.path.insert(0, str(source_dir.resolve()))
        from lightrag import QueryParam
        from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
        from lightrag.kg.shared_storage import initialize_share_data
        from lightrag.llm.openai import openai_embed
        from lightrag.namespace import NameSpace
        from lightrag.operate import naive_query
        from lightrag.utils import EmbeddingFunc, TiktokenTokenizer

        async def no_generation(*_args: Any, **_kwargs: Any) -> str:
            raise RuntimeError("The dense MCP tool must not generate an answer")

        async def embed_texts(texts: list[str], **kwargs: Any) -> Any:
            # Storage passes a scheduling hint that the raw embedding API does not accept.
            kwargs.pop("_priority", None)
            return await openai_embed.func(
                texts,
                model=os.environ.get("EMBEDDING_MODEL", "text-embedding-v4"),
                base_url=base_url,
                api_key=api_key,
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
            "llm_model_func": no_generation,
        }
        initialize_share_data()
        chunks_vdb = NanoVectorDBStorage(
            namespace=NameSpace.VECTOR_STORE_CHUNKS,
            workspace="",
            global_config=config,
            embedding_func=embedding,
            meta_fields={"full_doc_id", "content", "file_path"},
        )
        await chunks_vdb.initialize()
        _chunks_vdb = chunks_vdb
        _query_param_type = QueryParam
        _naive_query = naive_query
        _global_config = config
        return chunks_vdb, naive_query, config


@mcp.tool()
async def search_dense(
    query: str, chunk_top_k: int = 10, max_total_tokens: int = 20000
) -> dict[str, Any]:
    """Retrieve source-grounded text chunks by dense vector search."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if chunk_top_k < 1 or max_total_tokens < 1:
        raise ValueError("chunk_top_k and max_total_tokens must be positive")
    chunks_vdb, naive_query, config = await get_retriever()
    param = _query_param_type(
        mode="naive",
        chunk_top_k=chunk_top_k,
        max_total_tokens=max_total_tokens,
        enable_rerank=False,
        only_need_context=True,
    )
    result = await naive_query(query.strip(), chunks_vdb, param, config)
    payload = result.raw_data if result is not None else {
        "status": "failure",
        "message": "No relevant document chunks found.",
        "data": {},
        "metadata": {"failure_reason": "no_results", "mode": "naive"},
    }
    if not isinstance(payload, dict):
        raise RuntimeError("LightRAG returned an invalid response")
    return payload


if __name__ == "__main__":
    mcp.run(transport="stdio")
