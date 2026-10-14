"""Run the dense, LinearRAG, and LightRAG local retrieval cascade."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

from agentic_rag.graph import build_agent_graph
from agentic_rag.llm import GroundedLLM
from agentic_rag.tools.dense import MCPDenseRetriever
from agentic_rag.tools.linear import MCPLinearRetriever
from agentic_rag.tools.light import MCPLocalRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLM_ENV_FILE = PROJECT_ROOT / ".env"
_FILE_SETTINGS = dotenv_values(DEFAULT_LLM_ENV_FILE)


def setting(name: str, fallback: str = "") -> str:
    """Use a process setting first, then the local, untracked .env file."""
    return os.environ.get(name) or _FILE_SETTINGS.get(name) or fallback


LIGHTRAG_ROOT = Path(setting("LIGHTRAG_SOURCE_DIR", str(PROJECT_ROOT / "external" / "LightRAG")))
LIGHTRAG_INDEX_ROOT = Path(setting("LIGHTRAG_INDEX_ROOT", str(PROJECT_ROOT / "indexes" / "lightrag")))
DEFAULT_INDEX = Path(setting("LIGHTRAG_INDEX_DIR", str(LIGHTRAG_INDEX_ROOT / "hotpotqa" / "index")))
DEFAULT_ENV_FILE = Path(setting("LIGHTRAG_ENV_FILE", str(LIGHTRAG_ROOT / ".env")))
DEFAULT_LIGHTRAG_PYTHON = setting("LIGHTRAG_PYTHON", sys.executable)
LINEARRAG_ROOT = Path(setting("LINEARRAG_ROOT", str(PROJECT_ROOT / "external" / "LinearRAG")))
DEFAULT_LINEAR_PYTHON = setting("LINEARRAG_PYTHON", sys.executable)
DEFAULT_LINEAR_CACHE = Path(setting("LINEARRAG_CACHE_DIR", str(PROJECT_ROOT / "cache" / "linearrag")))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Run Dense Retrieval → LinearRAG → LightRAG local as needed")
    parser.add_argument("query", help="Question to retrieve evidence for")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--lightrag-root", type=Path, default=LIGHTRAG_ROOT)
    parser.add_argument("--lightrag-python", default=DEFAULT_LIGHTRAG_PYTHON)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--top-k", type=int, default=int(setting("LIGHTRAG_CHUNK_TOP_K", "10")))
    parser.add_argument("--max-total-tokens", type=int, default=int(setting("LIGHTRAG_MAX_TOTAL_TOKENS", "20000")))
    parser.add_argument("--linear-root", type=Path, default=LINEARRAG_ROOT)
    parser.add_argument("--linear-python", default=DEFAULT_LINEAR_PYTHON)
    parser.add_argument("--linear-cache", type=Path, default=DEFAULT_LINEAR_CACHE)
    parser.add_argument("--linear-dataset", default=setting("LINEARRAG_DATASET", "hotpotqa"))
    parser.add_argument("--linear-top-k", type=int, default=int(setting("LINEARRAG_TOP_K", "10")))
    parser.add_argument("--local-top-k", type=int, default=int(setting("LIGHTRAG_LOCAL_TOP_K", "40")))
    parser.add_argument("--local-chunk-top-k", type=int, default=int(setting("LIGHTRAG_LOCAL_CHUNK_TOP_K", "10")))
    parser.add_argument("--local-max-total-tokens", type=int, default=int(setting("LIGHTRAG_LOCAL_MAX_TOTAL_TOKENS", "30000")))
    parser.add_argument("--max-stage", choices=["dense", "linear", "light"], default="light")
    parser.add_argument("--llm-env-file", type=Path, default=DEFAULT_LLM_ENV_FILE, help="Project LLM configuration file")
    parser.add_argument("--llm-model", default=None, help="Override LLM_MODEL from the project .env file")
    parser.add_argument("--retrieve-only", action="store_true", help="Skip Judge and answer; show retrieval data")
    parser.add_argument("--linear-only", action="store_true", help="Test LinearRAG MCP retrieval without Judge or answer")
    parser.add_argument("--local-only", action="store_true", help="Test LightRAG local MCP retrieval without Judge or answer")
    parser.add_argument("--show-evidence", action="store_true", help="Print excerpts from every retrieval stage")
    parser.add_argument("--json", action="store_true", help="Print the complete LangGraph State as JSON")
    args = parser.parse_args()

    retriever = MCPDenseRetriever(
        server_python=args.lightrag_python,
        source_dir=args.lightrag_root,
        index_dir=args.index,
        env_file=args.env_file,
        chunk_top_k=args.top_k,
        max_total_tokens=args.max_total_tokens,
    )
    linear = MCPLinearRetriever(
        linearrag_python=args.linear_python,
        linearrag_root=args.linear_root,
        cache_dir=args.linear_cache,
        dataset=args.linear_dataset,
        top_k=args.linear_top_k,
    )
    local = MCPLocalRetriever(
        server_python=args.lightrag_python,
        source_dir=args.lightrag_root,
        index_dir=args.index,
        env_file=args.env_file,
        llm_env_file=args.llm_env_file,
        llm_model=args.llm_model,
        top_k=args.local_top_k,
        chunk_top_k=args.local_chunk_top_k,
        max_total_tokens=args.local_max_total_tokens,
    )
    if args.linear_only:
        evidence = asyncio.run(linear.search(args.query))
        print(json.dumps({"current_stage": "linear", "current_evidence": evidence}, ensure_ascii=False, indent=2))
        return
    if args.local_only:
        evidence = asyncio.run(local.search(args.query))
        print(json.dumps({"current_stage": "light", "current_evidence": evidence}, ensure_ascii=False, indent=2))
        return
    llm = None if args.retrieve_only else GroundedLLM(env_file=args.llm_env_file, model=args.llm_model)
    graph = build_agent_graph(
        retriever,
        llm,
        linear if args.max_stage in ("linear", "light") else None,
        local if args.max_stage == "light" else None,
    )
    state = asyncio.run(graph.ainvoke({"query": args.query}))
    if args.json or args.retrieve_only:
        print(json.dumps(state, ensure_ascii=False, indent=2))
    else:
        stage_names = {"dense": "第一阶段 Dense Retrieval", "linear": "第二阶段 LinearRAG", "light": "第三阶段 LightRAG local"}
        print("检索过程：")
        for event in state["stage_history"]:
            stage_name = stage_names.get(event["stage"], event["stage"])
            if event.get("tool"):
                evidence_label = "条证据（实体、关系和片段）" if event["stage"] == "light" else "条片段"
                print(f"{stage_name}：检索到 {event['evidence_count']} {evidence_label}，耗时 {event['elapsed_seconds']} 秒")
                if event.get("error"):
                    print(f"  检索错误：{event['error']}")
            elif event.get("node") == "judge":
                verdict = "证据充分" if event["sufficient"] else "证据不足"
                print(f"  Judge：{verdict}。{event.get('reason', '')}")
            elif event.get("node") == "compress":
                print(f"  Evidence Pool：保留 {event['kept_count']} 条{stage_name}证据")
        print("\n回答：")
        print(state["answer"])
        if state["answer_source_ids"]:
            print("\n来源：")
            cited_evidence = state["answer_evidence"]
            for number, item in enumerate(cited_evidence, start=1):
                if item["source_id"] in state["answer_source_ids"]:
                    print(f"[{number}] {item['source_ref']} (source_id={item['source_id']})")
        if args.show_evidence:
            for stage, evidence in state["stage_evidence"].items():
                print(f"\n{stage_names.get(stage, stage)}原始检索片段（{len(evidence)} 条）：")
                for item in evidence:
                    print(f"来源：{item['source_ref']} (source_id={item['source_id']})")
                    print(f"{item['content'][:300]}\n")
    if state["retrieval_error"] or state["judge_error"] or state["answer_error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
