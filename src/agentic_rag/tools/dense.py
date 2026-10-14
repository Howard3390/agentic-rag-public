"""MCP client for dense chunk retrieval backed by LightRAG."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from agentic_rag.state import Evidence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SERVER_SCRIPT = PROJECT_ROOT / "mcp_servers" / "lightrag_dense.py"


class RetrievalError(RuntimeError):
    """Raised when the MCP server or its LightRAG response is unusable."""


def normalize_chunks(payload: dict[str, Any]) -> list[Evidence]:
    if payload.get("status") != "success":
        raise RetrievalError(str(payload.get("message") or "LightRAG retrieval failed"))
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("chunks"), list):
        raise RetrievalError("LightRAG response has no data.chunks list")
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise RetrievalError("LightRAG response has invalid metadata")
    mode = metadata.get("query_mode")
    if mode is not None and mode != "naive":
        raise RetrievalError(f"Unexpected LightRAG mode: {mode}")

    evidence: list[Evidence] = []
    for chunk in data["chunks"]:
        if not isinstance(chunk, dict):
            continue
        content = chunk.get("content")
        chunk_id = chunk.get("chunk_id")
        file_path = chunk.get("file_path")
        if not all(isinstance(value, str) and value.strip() for value in (content, chunk_id, file_path)):
            raise RetrievalError("A retrieved chunk lacks content, chunk_id, or file_path")
        item: Evidence = {
            "evidence_id": f"dense:{chunk_id}",
            "content": content,
            "source_id": chunk_id,
            "source_ref": file_path,
            "retrieval_stage": "dense",
        }
        if chunk.get("reference_id") is not None:
            item["reference_id"] = str(chunk["reference_id"])
        evidence.append(item)
    return evidence


class MCPDenseRetriever:
    def __init__(
        self,
        *,
        server_python: str | Path | None = None,
        server_script: str | Path = DEFAULT_SERVER_SCRIPT,
        source_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
        env_file: str | Path | None = None,
        chunk_top_k: int = 10,
        max_total_tokens: int = 20000,
    ) -> None:
        self.server_python = str(server_python or os.getenv("LIGHTRAG_PYTHON") or sys.executable)
        self.server_script = Path(server_script)
        self.source_dir = str(source_dir or os.getenv("LIGHTRAG_SOURCE_DIR", ""))
        self.index_dir = str(index_dir or os.getenv("LIGHTRAG_INDEX_DIR", ""))
        self.env_file = str(env_file or os.getenv("LIGHTRAG_ENV_FILE", ""))
        self.chunk_top_k = chunk_top_k
        self.max_total_tokens = max_total_tokens

    async def search(self, query: str) -> list[Evidence]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not self.index_dir:
            raise ValueError("Set LIGHTRAG_INDEX_DIR or pass index_dir")
        if not self.source_dir:
            raise ValueError("Set LIGHTRAG_SOURCE_DIR or pass source_dir")

        child_env = {
            "LIGHTRAG_SOURCE_DIR": self.source_dir,
            "LIGHTRAG_INDEX_DIR": self.index_dir,
        }
        if self.env_file:
            child_env["LIGHTRAG_ENV_FILE"] = self.env_file
        params = StdioServerParameters(
            command=self.server_python,
            args=[str(self.server_script)],
            env=child_env,
        )
        try:
            async with Client(params, read_timeout_seconds=180) as client:
                result = await client.call_tool(
                    "search_dense",
                    {
                        "query": query,
                        "chunk_top_k": self.chunk_top_k,
                        "max_total_tokens": self.max_total_tokens,
                    },
                )
        except Exception as exc:
            raise RetrievalError(f"LightRAG MCP call failed: {exc}") from exc
        if result.is_error:
            message = " ".join(getattr(part, "text", "") for part in result.content).strip()
            raise RetrievalError(message or "LightRAG MCP tool returned an error")
        payload = result.structured_content
        if not isinstance(payload, dict):
            texts = [part.text for part in result.content if getattr(part, "type", None) == "text"]
            try:
                payload = json.loads(texts[0]) if len(texts) == 1 else None
            except json.JSONDecodeError as exc:
                raise RetrievalError("LightRAG MCP tool returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RetrievalError("LightRAG MCP tool returned no structured data")
        return normalize_chunks(payload)
