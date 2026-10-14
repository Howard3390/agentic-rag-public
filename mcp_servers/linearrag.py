"""MCP bridge to the separate Python 3.9 LinearRAG retrieval worker."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from mcp.server import MCPServer


mcp = MCPServer("linearrag-retrieval")
WORKER = Path(__file__).with_name("linearrag_worker.py")


@mcp.tool()
async def search_linear(query: str, dataset: str = "hotpotqa", top_k: int = 10) -> dict[str, Any]:
    """Retrieve source passages with LinearRAG's graph retrieval; do not generate an answer."""
    if not query.strip() or top_k < 1:
        raise ValueError("query must be nonempty and top_k positive")
    root = Path(os.environ["LINEARRAG_ROOT"])
    python = os.environ["LINEARRAG_PYTHON"]
    cache_dir = Path(os.environ["LINEARRAG_CACHE_DIR"]) / dataset
    request = {
        "query": query,
        "dataset": dataset,
        "top_k": top_k,
        "repo_root": str(root),
        "cache_dir": str(cache_dir),
    }
    try:
        process = await asyncio.create_subprocess_exec(
            python,
            str(WORKER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except OSError as exc:
        return {"status": "error", "error": f"无法启动 LinearRAG 子进程：{type(exc).__name__}: {exc}"}
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(json.dumps(request, ensure_ascii=False).encode("utf-8")),
            timeout=180,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return {"status": "error", "error": "LinearRAG 检索超过 180 秒，已停止子进程"}
    try:
        payload = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"status": "error", "error": "LinearRAG 子进程未返回有效 JSON，请查看启动终端的错误日志"}
    if not isinstance(payload, dict):
        return {"status": "error", "error": "LinearRAG 子进程返回的数据格式无效"}
    if process.returncode != 0 or payload.get("status") != "success":
        error = str(payload.get("error") or "LinearRAG 子进程执行失败")
        if error.strip() == "MemoryError:" or error.startswith("ArrowMemoryError:"):
            error = f"LinearRAG 加载索引时内存分配失败（{error.strip()}）；请检查 Windows 虚拟内存和页面文件"
        return {"status": "error", "error": error}
    return payload


if __name__ == "__main__":
    mcp.run(transport="stdio")
