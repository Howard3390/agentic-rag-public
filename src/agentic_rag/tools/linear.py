"""MCP client for read-only LinearRAG passage retrieval."""

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
DEFAULT_SERVER_SCRIPT = PROJECT_ROOT / "mcp_servers" / "linearrag.py"


def normalize_passages(payload: dict[str, Any]) -> list[Evidence]:
    if payload.get("status") != "success" or not isinstance(payload.get("passages"), list):
        raise RetrievalError(str(payload.get("error") or "LinearRAG retrieval failed"))
    evidence: list[Evidence] = []
    for passage in payload["passages"]:
        if not isinstance(passage, dict):
            raise RetrievalError("LinearRAG passage is invalid")
        content = passage.get("content")
        source_id = passage.get("source_id")
        source_ref = passage.get("source_ref")
        if not all(isinstance(value, str) and value.strip() for value in (content, source_id, source_ref)):
            raise RetrievalError("LinearRAG passage lacks content or source")
        item: Evidence = {
            "evidence_id": f"linear:{source_id}",
            "content": content,
            "source_id": source_id,
            "source_ref": source_ref + f"#chunk={passage.get('chunk_index')}",
            "retrieval_stage": "linear",
        }
        if passage.get("chunk_index") is not None:
            item["reference_id"] = str(passage["chunk_index"])
        evidence.append(item)
    return evidence


class MCPLinearRetriever:
    def __init__(
        self,
        *,
        linearrag_python: str | Path,
        linearrag_root: str | Path,
        cache_dir: str | Path,
        dataset: str = "hotpotqa",
        top_k: int = 10,
        server_python: str | Path | None = None,
        server_script: str | Path = DEFAULT_SERVER_SCRIPT,
    ) -> None:
        self.linearrag_python = str(linearrag_python)
        self.linearrag_root = str(linearrag_root)
        self.cache_dir = str(cache_dir)
        self.dataset = dataset
        self.top_k = top_k
        self.server_python = str(server_python or sys.executable)
        self.server_script = Path(server_script)

    async def search(self, query: str) -> list[Evidence]:
        if not query.strip():
            raise ValueError("query must not be empty")
        params = StdioServerParameters(
            command=self.server_python,
            args=[str(self.server_script)],
            env={
                "LINEARRAG_ROOT": self.linearrag_root,
                "LINEARRAG_PYTHON": self.linearrag_python,
                "LINEARRAG_CACHE_DIR": self.cache_dir,
            },
        )
        try:
            async with Client(params, read_timeout_seconds=200) as client:
                result = await client.call_tool(
                    "search_linear",
                    {"query": query, "dataset": self.dataset, "top_k": self.top_k},
                )
        except Exception as exc:
            raise RetrievalError(f"LinearRAG MCP call failed: {exc}") from exc
        if result.is_error:
            message = " ".join(getattr(part, "text", "") for part in result.content).strip()
            raise RetrievalError(message or "LinearRAG MCP tool returned an error")
        payload = result.structured_content
        if not isinstance(payload, dict):
            texts = [part.text for part in result.content if getattr(part, "type", None) == "text"]
            try:
                payload = json.loads(texts[0]) if len(texts) == 1 else None
            except json.JSONDecodeError as exc:
                raise RetrievalError("LinearRAG MCP tool returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RetrievalError("LinearRAG MCP tool returned no structured data")
        return normalize_passages(payload)
