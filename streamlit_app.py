"""Local Streamlit interface for the Agentic RAG cascade."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import streamlit as st

from agentic_rag.cli import (
    DEFAULT_ENV_FILE,
    DEFAULT_LIGHTRAG_PYTHON,
    DEFAULT_LINEAR_CACHE,
    DEFAULT_LINEAR_PYTHON,
    DEFAULT_LLM_ENV_FILE,
    LIGHTRAG_INDEX_ROOT,
    LIGHTRAG_ROOT,
    LINEARRAG_ROOT,
)
from agentic_rag.graph import build_agent_graph
from agentic_rag.llm import GroundedLLM
from agentic_rag.tools.dense import MCPDenseRetriever
from agentic_rag.tools.light import MCPLocalRetriever
from agentic_rag.tools.linear import MCPLinearRetriever


DATASETS = {
    "HotpotQA": ("hotpotqa", "hotpotqa"),
    "2WikiMultiHopQA": ("2wikimultihopqa", "2wikimultihop"),
    "MuSiQue": ("musique", "musique"),
}
STAGES = {
    "完整流程 · 三阶段": "light",
    "运行至 LinearRAG": "linear",
    "仅 Dense Retrieval": "dense",
}
STAGE_NAMES = {
    "dense": "Dense Retrieval",
    "linear": "LinearRAG",
    "light": "LightRAG local",
}


def index_for(dataset: str) -> Path:
    return LIGHTRAG_INDEX_ROOT / dataset / "index"


def build_graph(dataset_name: str, max_stage: str):
    lightrag_dataset, linear_dataset = DATASETS[dataset_name]
    index_dir = index_for(lightrag_dataset)
    if not index_dir.is_dir():
        raise FileNotFoundError(f"未找到 {dataset_name} 的 LightRAG 索引：{index_dir}")
    if max_stage != "dense" and not (LINEARRAG_ROOT / "import" / linear_dataset / "ner_results.json").is_file():
        raise FileNotFoundError(f"未找到 {dataset_name} 的 LinearRAG 索引")
    if not DEFAULT_LLM_ENV_FILE.is_file():
        raise FileNotFoundError("未找到项目 LLM 配置文件 .env")

    dense = MCPDenseRetriever(
        server_python=DEFAULT_LIGHTRAG_PYTHON,
        source_dir=LIGHTRAG_ROOT,
        index_dir=index_dir,
        env_file=DEFAULT_ENV_FILE,
        chunk_top_k=10,
        max_total_tokens=20000,
    )
    linear = None
    if max_stage in ("linear", "light"):
        linear = MCPLinearRetriever(
            linearrag_python=DEFAULT_LINEAR_PYTHON,
            linearrag_root=LINEARRAG_ROOT,
            cache_dir=DEFAULT_LINEAR_CACHE,
            dataset=linear_dataset,
            top_k=10,
        )
    local = None
    if max_stage == "light":
        local = MCPLocalRetriever(
            server_python=DEFAULT_LIGHTRAG_PYTHON,
            source_dir=LIGHTRAG_ROOT,
            index_dir=index_dir,
            env_file=DEFAULT_ENV_FILE,
            llm_env_file=DEFAULT_LLM_ENV_FILE,
            top_k=40,
            chunk_top_k=10,
            max_total_tokens=30000,
        )
    llm = GroundedLLM(env_file=DEFAULT_LLM_ENV_FILE)
    return build_agent_graph(dense, llm, linear, local), llm


def needs_translation(query: str) -> bool:
    return bool(re.search(r"[A-Za-z]", query)) and not bool(re.search(r"[\u4e00-\u9fff]", query))


def event_text(event: dict[str, Any]) -> str:
    stage = STAGE_NAMES.get(event.get("stage"), str(event.get("stage", "")))
    if event.get("tool"):
        if event.get("error"):
            return f"⚠️ {stage} 检索失败：{event['error']}"
        unit = "条证据" if event.get("stage") == "light" else "条片段"
        return f"🔎 {stage}：找到 {event['evidence_count']} {unit} · {event['elapsed_seconds']} 秒"
    if event.get("node") == "judge":
        verdict = "证据充分" if event.get("sufficient") else "证据不足，继续检索"
        return f"{'✅' if event.get('sufficient') else '↪️'} Judge：{verdict}。判断依据见下方。"
    if event.get("node") == "compress":
        return f"📌 Evidence Pool：保留 {event.get('kept_count', 0)} 条可复用证据"
    if event.get("node") == "answer":
        return "✍️ 已生成回答" if not event.get("error") else f"⚠️ 回答生成失败：{event['error']}"
    return ""


async def run_graph(graph: Any, query: str, status: Any) -> dict[str, Any]:
    final_state: dict[str, Any] = {}
    seen_events = 0
    async for state in graph.astream({"query": query}, stream_mode="values"):
        final_state = state
        history = state.get("stage_history") or []
        for event in history[seen_events:]:
            message = event_text(event)
            if message:
                status.write(message)
            if event.get("tool"):
                status.update(label=f"正在分析 {STAGE_NAMES.get(event['stage'], event['stage'])} 的结果…")
        seen_events = len(history)
    return final_state


async def run_query(graph: Any, llm: GroundedLLM, query: str, status: Any):
    translation_task = asyncio.create_task(llm.translate_question(query)) if needs_translation(query) else None
    state = await run_graph(graph, query, status)
    judge_events = [
        (index, event) for index, event in enumerate(state.get("stage_history") or [])
        if event.get("node") == "judge"
    ]
    judge_display = {}
    if judge_events:
        try:
            translated = await llm.translate_judgements([event for _, event in judge_events])
            judge_display = {index: item for (index, _), item in zip(judge_events, translated)}
            for item in translated:
                if item["reason"]:
                    status.write(f"判断依据：{item['reason']}")
        except Exception:
            status.write("⚠️ 判断依据中文翻译暂不可用，原始记录可在下方下载。")
    translated_query = None
    if translation_task is not None:
        try:
            translated_query = await translation_task
        except Exception:
            status.write("⚠️ 问题翻译暂不可用，检索与回答已按原问题完成。")
    return state, translated_query, judge_display


def evidence_kind(item: dict[str, Any]) -> str:
    source_id = str(item.get("source_id", ""))
    if source_id.startswith("entity:"):
        return "图实体"
    if source_id.startswith("relationship:"):
        return "图关系"
    return "原文片段"


def show_result(run: dict[str, Any]) -> None:
    state = run["state"]
    history = state.get("stage_history") or []
    judge_display = run.get("judge_display") or {}
    retrieved = [event for event in history if event.get("tool")]
    errors = [event for event in history if event.get("error")]

    st.divider()
    st.subheader("回答")
    st.caption(f"原问题：{run['query']}  ·  数据集：{run['dataset']}")
    if run.get("translated_query"):
        st.write(f"**中文问题：** {run['translated_query']}")
    elif needs_translation(run["query"]):
        st.caption("中文翻译暂不可用。")
    if state.get("sufficient") and not state.get("answer_error"):
        st.markdown(state.get("answer") or "未生成回答")
    else:
        answer = state.get("answer") or "证据不足，未生成回答。"
        if not state.get("sufficient"):
            last_judge_index = next(
                (index for index in range(len(history) - 1, -1, -1) if history[index].get("node") == "judge"),
                None,
            )
            translated = judge_display.get(last_judge_index) if last_judge_index is not None else None
            missing = translated.get("missing_info") if translated else []
            answer = "当前已接入的检索阶段证据不足，无法可靠回答。"
            if missing:
                answer += " 缺少：" + "；".join(missing) + "。"
        st.warning(answer)
    if errors:
        st.caption(f"本次有 {len(errors)} 个阶段事件报错，详情见下方检索过程。")

    columns = st.columns(4)
    columns[0].metric("停止阶段", STAGE_NAMES.get(state.get("current_stage"), "—"))
    columns[1].metric("检索阶段", str(len(retrieved)))
    columns[2].metric("证据池", f"{len(state.get('evidence_pool') or [])} 条")
    columns[3].metric("总耗时", f"{run['elapsed_seconds']:.1f} 秒")

    tabs = st.tabs(["引用来源", "检索过程", "各阶段证据"])
    with tabs[0]:
        cited = set(state.get("answer_source_ids") or [])
        evidence = state.get("answer_evidence") or []
        if not cited:
            st.info("这次回答没有可展示的引用来源。")
        for number, item in enumerate(evidence, start=1):
            if item["source_id"] not in cited:
                continue
            with st.expander(f"[{number}] {STAGE_NAMES.get(item['retrieval_stage'], '')} · {item['source_id']}"):
                st.caption(f"来源：{item['source_ref']}")
                st.write(item["content"])

    with tabs[1]:
        rows = []
        for event in history:
            kind = "检索" if event.get("tool") else {"judge": "判断", "compress": "保留证据", "answer": "回答"}.get(event.get("node"), "")
            if not kind:
                continue
            outcome = (
                f"{event.get('evidence_count', 0)} 条证据"
                if kind == "检索"
                else ("充分" if event.get("sufficient") else "不足")
                if kind == "判断"
                else f"保留 {event.get('kept_count', 0)} 条"
                if kind == "保留证据"
                else "完成"
            )
            rows.append({
                "阶段": STAGE_NAMES.get(event.get("stage"), ""),
                "步骤": kind,
                "结果": outcome,
                "耗时（秒）": event.get("elapsed_seconds"),
            })
        st.dataframe(rows, hide_index=True, width="stretch")
        for index, event in enumerate(history):
            if event.get("node") == "judge":
                with st.expander(f"{STAGE_NAMES.get(event['stage'])} · Judge 判断依据"):
                    translated = judge_display.get(index)
                    reason = translated.get("reason") if translated else None
                    st.write(reason or "中文判断依据暂不可用，原始内容可从下方运行记录下载。")
                    if event.get("missing_info"):
                        missing = translated.get("missing_info") if translated else None
                        st.write("缺少：" + "；".join(missing) if missing else "缺少信息的中文翻译暂不可用。")
                    if event.get("error"):
                        st.error(event["error"])
            elif event.get("error"):
                st.error(f"{STAGE_NAMES.get(event['stage'])}：{event['error']}")

    with tabs[2]:
        stage_evidence = state.get("stage_evidence") or {}
        if not stage_evidence:
            st.info("没有检索到可展示的证据。")
        else:
            stage = st.selectbox("选择阶段", list(stage_evidence), format_func=lambda value: STAGE_NAMES[value])
            items = stage_evidence[stage]
            kinds = ["原文片段"]
            if stage == "light":
                kinds += ["图实体", "图关系"]
            kind = st.radio("证据类型", kinds, horizontal=True)
            selected = [item for item in items if evidence_kind(item) == kind]
            st.caption(f"共 {len(selected)} 条；下方展示前 {min(len(selected), 15)} 条。图实体和关系不计入 10 条原文片段上限。")
            for item in selected[:15]:
                with st.expander(item["source_id"]):
                    st.caption(f"来源：{item['source_ref']}")
                    st.write(item["content"])

    with st.expander("下载本次运行记录"):
        st.download_button(
            "下载 JSON",
            data=json.dumps({**state, "translated_query": run.get("translated_query"), "judge_display": judge_display}, ensure_ascii=False, indent=2),
            file_name="agentic_rag_result.json",
            mime="application/json",
        )


def main() -> None:
    st.set_page_config(page_title="Agentic RAG 问答台", page_icon="🔎", layout="wide")
    st.title("Agentic RAG 问答台")
    st.caption("按顺序检索证据，证据足够时提前回答。")
    st.markdown(
        "**三个检索阶段各有什么特点？**\n\n"
        "- **Dense Retrieval**：用向量相似度检索原文片段，适合快速找到与问题直接相关的内容。\n"
        "- **LinearRAG**：从问题中的实体出发扩展关联并排序段落，适合串联跨文档线索。\n"
        "- **LightRAG local**：结合知识图谱中的实体、关系和原文，适合补充较复杂的关联证据。"
    )

    with st.sidebar:
        st.header("检索设置")
        dataset = st.selectbox("数据集", list(DATASETS))
        stage_choice = st.selectbox("运行范围", list(STAGES))
        st.caption("每阶段最多返回 10 条原文片段；LightRAG local 还会返回图实体和关系。")
        st.divider()
        st.caption("Dense Retrieval → LinearRAG → LightRAG local")

    with st.form("question_form"):
        query = st.text_area("输入问题", placeholder="请输入需要检索的问题…", height=100)
        submitted = st.form_submit_button("开始检索并回答", type="primary", width="stretch")

    if submitted:
        query = query.strip()
        if not query:
            st.warning("请先输入问题。")
        else:
            try:
                graph, llm = build_graph(dataset, STAGES[stage_choice])
                started = perf_counter()
                with st.status("正在检索证据…", expanded=True) as status:
                    state, translated_query, judge_display = asyncio.run(run_query(graph, llm, query, status))
                    has_errors = any(event.get("error") for event in state.get("stage_history", []))
                    status.update(
                        label="检索结束（存在错误）" if has_errors else "检索完成",
                        state="error" if has_errors else "complete",
                        expanded=has_errors,
                    )
                st.session_state["latest_run"] = {
                    "query": query,
                    "translated_query": translated_query,
                    "judge_display": judge_display,
                    "dataset": dataset,
                    "state": state,
                    "elapsed_seconds": perf_counter() - started,
                }
            except Exception as exc:
                st.error(f"运行失败：{exc}")

    if "latest_run" in st.session_state:
        show_result(st.session_state["latest_run"])
    elif not submitted:
        st.info("选择数据集并输入问题，即可查看回答、阶段进展和证据来源。")


if __name__ == "__main__":
    main()
