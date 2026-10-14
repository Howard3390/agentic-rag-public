"""Small MCP server used to verify the graph's real stdio transport."""

from mcp.server import MCPServer


mcp = MCPServer("fake-lightrag-dense")


@mcp.tool()
def search_dense(query: str, chunk_top_k: int = 10, max_total_tokens: int = 20000) -> dict:
    """Return a source-grounded retrieval fixture."""
    return {
        "status": "success",
        "message": "ok",
        "data": {
            "entities": [],
            "relationships": [],
            "chunks": [
                {
                    "content": f"Evidence for {query}",
                    "chunk_id": "chunk-001",
                    "file_path": "hotpotqa/doc-001",
                    "reference_id": "1",
                }
            ][:chunk_top_k],
            "references": [{"reference_id": "1", "file_path": "hotpotqa/doc-001"}],
        },
        "metadata": {"query_mode": "naive"},
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
