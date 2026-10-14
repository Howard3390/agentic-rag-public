"""MCP client for LightRAG local knowledge-graph retrieval."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from agentic_rag.state import Evidence
from agentic_rag.tools.dense import RetrievalError


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SERVER_SCRIPT = PROJECT_ROOT / "mcp_servers" / "lightrag_local.py"


def normalize_local_chunks(payload: dict[str, Any]) -> list[Evidence]:
    if payload.get("status") != "success":
        raise RetrievalError(str(payload.get("message") or "LightRAG local retrieval failed"))
    data = payload.get("data")
    metadata = payload.get("metadata")
    if not isinstance(data, dict) or any(not isinstance(data.get(key), list) for key in ("entities", "relationships", "chunks")):
        raise RetrievalError("LightRAG local response lacks entity, relationship, or chunk data")
    if not isinstance(metadata, dict) or metadata.get("query_mode") != "local":
        raise RetrievalError("LightRAG response is not from local mode")
    evidence: list[Evidence] = []
    for entity in data["entities"]:
        if not isinstance(entity, dict):
            raise RetrievalError("LightRAG local response contains an invalid entity")
        name = entity.get("entity_name")
        description = entity.get("description")
        if not isinstance(name, str) or not name.strip() or not isinstance(description, str) or not description.strip():
            continue
        item: Evidence = {
            "evidence_id": f"light:entity:{name}",
            "content": f"Entity: {name}\nDescription: {description}",
            "source_id": f"entity:{name}",
            "source_ref": str(entity.get("file_path") or "LightRAG graph"),
            "retrieval_stage": "light",
        }
        if entity.get("source_id"):
            item["reference_id"] = str(entity["source_id"])
        evidence.append(item)
    for relation in data["relationships"]:
        if not isinstance(relation, dict):
            raise RetrievalError("LightRAG local response contains an invalid relationship")
        source = relation.get("src_id")
        target = relation.get("tgt_id")
        description = relation.get("description")
        if not all(isinstance(value, str) and value.strip() for value in (source, target, description)):
            continue
        item: Evidence = {
            "evidence_id": f"light:relationship:{source}:{target}",
            "content": f"Relationship: {source} -> {target}\nDescription: {description}",
            "source_id": f"relationship:{source}:{target}",
            "source_ref": str(relation.get("file_path") or "LightRAG graph"),
            "retrieval_stage": "light",
        }
        if relation.get("source_id"):
            item["reference_id"] = str(relation["source_id"])
        evidence.append(item)
    for chunk in data["chunks"]:
        if not isinstance(chunk, dict):
            raise RetrievalError("LightRAG local response contains an invalid chunk")
        content = chunk.get("content")
        chunk_id = chunk.get("chunk_id")
        file_path = chunk.get("file_path")
        if not all(isinstance(value, str) and value.strip() for value in (content, chunk_id, file_path)):
            raise RetrievalError("A local-mode chunk lacks content, chunk_id, or file_path")
        item: Evidence = {
            "evidence_id": f"light:{chunk_id}",
            "content": content,
            "source_id": chunk_id,
            "source_ref": file_path,
            "retrieval_stage": "light",
        }
        if chunk.get("reference_id") is not None:
            item["reference_id"] = str(chunk["reference_id"])
        evidence.append(item)
    return evidence


class MCPLocalRetriever:
    def __init__(
        self,
        *,
        server_python: str | Path | None = None,
        server_script: str | Path = DEFAULT_SERVER_SCRIPT,
        source_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
        env_file: str | Path | None = None,
        llm_env_file: str | Path | None = None,
        llm_model: str | None = None,
        top_k: int = 40,
        chunk_top_k: int = 10,
        max_total_tokens: int = 30000,
    ) -> None:
        self.server_python = str(server_python or os.getenv("LIGHTRAG_PYTHON") or sys.executable)
        self.server_script = Path(server_script)
        self.source_dir = str(source_dir or os.getenv("LIGHTRAG_SOURCE_DIR", ""))
        self.index_dir = str(index_dir or os.getenv("LIGHTRAG_INDEX_DIR", ""))
        self.env_file = str(env_file or os.getenv("LIGHTRAG_ENV_FILE", ""))
        self.llm_env_file = str(llm_env_file or "")
        self.llm_model = llm_model
        self.top_k = top_k
        self.chunk_top_k = chunk_top_k
        self.max_total_tokens = max_total_tokens

    async def search(self, query: str) -> list[Evidence]:
        if not query.strip():
            raise ValueError("query must not be empty")
        child_env = {
            "LIGHTRAG_SOURCE_DIR": self.source_dir,
            "LIGHTRAG_INDEX_DIR": self.index_dir,
            "LIGHTRAG_ENV_FILE": self.env_file,
            "AGENTIC_RAG_LLM_ENV_FILE": self.llm_env_file,
        }
        if self.llm_model:
            child_env["AGENTIC_RAG_LLM_MODEL"] = self.llm_model
        params = StdioServerParameters(
            command=self.server_python,
            args=[str(self.server_script)],
            env=child_env,
        )
        try:
            async with Client(params, read_timeout_seconds=240) as client:
                result = await client.call_tool(
                    "search_local",
                    {
                        "query": query,
                        "top_k": self.top_k,
                        "chunk_top_k": self.chunk_top_k,
                        "max_total_tokens": self.max_total_tokens,
                    },
                )
        except Exception as exc:
            raise RetrievalError(f"LightRAG local MCP call failed: {exc}") from exc
        if result.is_error:
            message = " ".join(getattr(part, "text", "") for part in result.content).strip()
            raise RetrievalError(message or "LightRAG local MCP tool returned an error")
        payload = result.structured_content
        if not isinstance(payload, dict):
            texts = [part.text for part in result.content if getattr(part, "type", None) == "text"]
            try:
                payload = json.loads(texts[0]) if len(texts) == 1 else None
            except json.JSONDecodeError as exc:
                raise RetrievalError("LightRAG local MCP tool returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RetrievalError("LightRAG local MCP tool returned no structured data")
        return normalize_local_chunks(payload)
