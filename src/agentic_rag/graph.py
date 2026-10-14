"""LangGraph cascade: dense retrieval, LinearRAG, then LightRAG local."""

from __future__ import annotations

import re
from time import perf_counter
from typing import Protocol

from langgraph.graph import END, START, StateGraph

from agentic_rag.llm import GroundedLLM
from agentic_rag.state import AgentState, Evidence


class Retriever(Protocol):
    async def search(self, query: str) -> list[Evidence]: ...


def compress_evidence(query: str, evidence: list[Evidence], max_items: int = 4) -> list[Evidence]:
    """Keep short, exact excerpts from the most query-relevant retrieved passages."""
    stopwords = {"what", "which", "where", "when", "who", "whose", "with", "from", "that", "this", "were", "was", "are", "the", "and", "for", "year"}
    terms = [word.lower() for word in re.findall(r"\w+", query) if len(word) >= 3 and word.lower() not in stopwords]
    ranked = []
    for position, item in enumerate(evidence):
        content = item["content"]
        lowered = content.lower()
        hits = [term for term in terms if term in lowered]
        if not hits:
            continue
        score = sum(3 if term.isdigit() else 1 for term in hits)
        first = min(lowered.find(term) for term in hits)
        start = max(0, first - 200)
        excerpt = content[start:start + 1600]
        ranked.append((score, position, {**item, "content": excerpt}))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [item for _, _, item in ranked[:max_items]]


def build_agent_graph(
    dense_retriever: Retriever,
    llm: GroundedLLM | None = None,
    linear_retriever: Retriever | None = None,
    light_retriever: Retriever | None = None,
):
    if light_retriever is not None and linear_retriever is None:
        raise ValueError("LightRAG local requires the preceding LinearRAG stage")

    def initialize(state: AgentState) -> dict:
        if not state["query"].strip():
            raise ValueError("query must not be empty")
        return {
            "current_stage": "dense",
            "current_evidence": [],
            "stage_evidence": {},
            "evidence_pool": [],
            "stage_history": [],
            "retrieval_error": None,
            "sufficient": False,
            "judge_reason": "",
            "missing_info": [],
            "supporting_ids": [],
            "judge_error": None,
            "answer": None,
            "answer_evidence": [],
            "answer_source_ids": [],
            "answer_error": None,
        }

    async def retrieve_current(state: AgentState) -> dict:
        stage = state["current_stage"]
        retrievers = {"dense": dense_retriever, "linear": linear_retriever, "light": light_retriever}
        retriever = retrievers[stage]
        started = perf_counter()
        try:
            if retriever is None:
                raise RuntimeError(f"No retriever configured for stage {stage}")
            evidence = await retriever.search(state["query"])
            error = None
        except Exception as exc:
            evidence = []
            error = str(exc)
        return {
            "current_evidence": evidence,
            "stage_evidence": {**state["stage_evidence"], stage: evidence},
            "retrieval_error": error,
            "stage_history": state["stage_history"] + [{
                "stage": stage,
                "tool": {"dense": "lightrag_dense", "linear": "linearrag", "light": "lightrag_local"}[stage],
                "evidence_count": len(evidence),
                "elapsed_seconds": round(perf_counter() - started, 3),
                "error": error,
            }],
        }

    async def judge_evidence(state: AgentState) -> dict:
        started = perf_counter()
        stage = state["current_stage"]
        evidence = (state["current_evidence"] + state["evidence_pool"]
                    if stage == "light" else state["evidence_pool"] + state["current_evidence"])
        if state["retrieval_error"]:
            reason = f"{stage} 检索失败，无法判断证据是否足够"
            decision = None
            error = state["retrieval_error"]
        elif not evidence:
            reason = f"{stage} 阶段没有检索到片段"
            decision = None
            error = None
        else:
            try:
                decision = await llm.judge(state["query"], evidence)
                reason = decision.reason
                error = None
            except Exception as exc:
                decision = None
                reason = "LLM 判断失败"
                error = str(exc)
        return {
            "sufficient": decision.sufficient if decision else False,
            "judge_reason": reason,
            "missing_info": decision.missing_info if decision else ["需要可核对的充分证据"],
            "supporting_ids": decision.supporting_ids if decision else [],
            "judge_error": error,
            "stage_history": state["stage_history"] + [{
                "stage": stage,
                "node": "judge",
                "sufficient": decision.sufficient if decision else False,
                "reason": reason,
                "missing_info": decision.missing_info if decision else ["需要可核对的充分证据"],
                "elapsed_seconds": round(perf_counter() - started, 3),
                "error": error,
            }],
        }

    def route_after_judge(state: AgentState) -> str:
        if state["sufficient"]:
            return "generate_answer"
        if state["current_stage"] == "dense" and linear_retriever is not None:
            return "compress"
        if state["current_stage"] == "linear" and light_retriever is not None:
            return "compress"
        return "insufficient"

    def compress(state: AgentState) -> dict:
        selected = compress_evidence(state["query"], state["current_evidence"])
        known = {item["evidence_id"] for item in state["evidence_pool"]}
        added = [item for item in selected if item["evidence_id"] not in known]
        return {
            "evidence_pool": state["evidence_pool"] + added,
            "stage_history": state["stage_history"] + [{
                "stage": state["current_stage"], "node": "compress", "kept_count": len(added),
            }],
        }

    def advance_stage(state: AgentState) -> dict:
        return {
            "current_stage": "linear" if state["current_stage"] == "dense" else "light",
            "current_evidence": [],
            "retrieval_error": None,
            "sufficient": False,
            "judge_reason": "",
            "missing_info": [],
            "supporting_ids": [],
            "judge_error": None,
        }

    async def generate_answer(state: AgentState) -> dict:
        started = perf_counter()
        evidence = (state["current_evidence"] + state["evidence_pool"]
                    if state["current_stage"] == "light"
                    else state["evidence_pool"] + state["current_evidence"])
        selected = [evidence[number - 1] for number in state["supporting_ids"]
                    if 1 <= number <= len(evidence)]
        answer_evidence = selected or evidence
        try:
            answer, source_ids = await llm.answer(state["query"], answer_evidence)
            error = None
        except Exception as exc:
            answer = "模型未能生成带有效来源的回答。"
            source_ids = []
            error = str(exc)
        return {
            "answer": answer,
            "answer_evidence": answer_evidence,
            "answer_source_ids": source_ids,
            "answer_error": error,
            "stage_history": state["stage_history"] + [{
                "stage": state["current_stage"], "node": "answer",
                "elapsed_seconds": round(perf_counter() - started, 3), "error": error,
            }],
        }

    def insufficient(state: AgentState) -> dict:
        missing = "；".join(state["missing_info"])
        answer = "当前已接入的检索阶段证据不足，无法可靠回答。"
        if missing:
            answer += f" 缺少：{missing}。"
        return {"answer": answer, "answer_source_ids": []}

    builder = StateGraph(AgentState)
    builder.add_node("initialize", initialize)
    builder.add_node("retrieve_current", retrieve_current)
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "retrieve_current")
    if llm is None:
        builder.add_edge("retrieve_current", END)
    else:
        builder.add_node("judge_evidence", judge_evidence)
        builder.add_node("generate_answer", generate_answer)
        builder.add_node("insufficient", insufficient)
        builder.add_edge("retrieve_current", "judge_evidence")
        builder.add_conditional_edges("judge_evidence", route_after_judge)
        builder.add_edge("generate_answer", END)
        builder.add_edge("insufficient", END)
        if linear_retriever is not None:
            builder.add_node("compress", compress)
            builder.add_node("advance_stage", advance_stage)
            builder.add_edge("compress", "advance_stage")
            builder.add_edge("advance_stage", "retrieve_current")
    return builder.compile()


def build_first_stage_graph(retriever: Retriever, llm: GroundedLLM | None = None):
    """Compatibility helper for isolated first-stage tests and retrieval-only mode."""
    return build_agent_graph(retriever, llm)
