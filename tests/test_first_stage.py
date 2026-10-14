import asyncio
import sys
import unittest
from pathlib import Path

from agentic_rag.graph import build_agent_graph, build_first_stage_graph
from agentic_rag.llm import JudgeDecision, evidence_context
from agentic_rag.tools.dense import MCPDenseRetriever, RetrievalError, normalize_chunks
from agentic_rag.tools.light import normalize_local_chunks


FAKE_SERVER = Path(__file__).with_name("fake_mcp_server.py")


class FakeLLM:
    def __init__(self, sufficient: bool) -> None:
        self.sufficient = sufficient
        self.answer_calls = 0

    async def judge(self, query, evidence):
        return JudgeDecision(self.sufficient, "测试判断", [] if self.sufficient else ["出生年份"], [1] if self.sufficient else [])

    async def answer(self, query, evidence):
        self.answer_calls += 1
        return "测试回答。[1]", [evidence[0]["source_id"]]


class ScriptedRetriever:
    def __init__(self, evidence):
        self.evidence = evidence
        self.calls = 0

    async def search(self, query):
        self.calls += 1
        return self.evidence


class CascadeLLM:
    def __init__(self):
        self.judge_calls = 0

    async def judge(self, query, evidence):
        self.judge_calls += 1
        if self.judge_calls == 1:
            return JudgeDecision(False, "缺少出生年份", ["出生年份"], [])
        return JudgeDecision(True, "两条证据可以回答", [], [1, 2])

    async def answer(self, query, evidence):
        return "乔治五世出生于1865年。[1][2]", [item["source_id"] for item in evidence]


class FirstStageTests(unittest.TestCase):
    def make_retriever(self) -> MCPDenseRetriever:
        return MCPDenseRetriever(
            server_python=sys.executable,
            server_script=FAKE_SERVER,
            source_dir="fake-source",
            index_dir="fake-index",
        )

    def test_graph_uses_mcp_and_preserves_source(self) -> None:
        retriever = self.make_retriever()
        result = asyncio.run(build_first_stage_graph(retriever).ainvoke({"query": "who wrote it?"}))
        self.assertEqual(result["current_stage"], "dense")
        self.assertEqual(result["retrieval_error"], None)
        self.assertEqual(result["evidence_pool"], [])
        self.assertEqual(result["current_evidence"][0]["source_id"], "chunk-001")
        self.assertEqual(result["current_evidence"][0]["source_ref"], "hotpotqa/doc-001")
        self.assertEqual(result["stage_evidence"]["dense"][0]["source_id"], "chunk-001")

    def test_sufficient_evidence_routes_to_answer(self) -> None:
        llm = FakeLLM(True)
        result = asyncio.run(build_first_stage_graph(self.make_retriever(), llm).ainvoke({"query": "who wrote it?"}))
        self.assertTrue(result["sufficient"])
        self.assertEqual(result["answer"], "测试回答。[1]")
        self.assertEqual(result["answer_source_ids"], ["chunk-001"])
        self.assertEqual(llm.answer_calls, 1)

    def test_insufficient_evidence_skips_answer(self) -> None:
        llm = FakeLLM(False)
        result = asyncio.run(build_first_stage_graph(self.make_retriever(), llm).ainvoke({"query": "who wrote it?"}))
        self.assertFalse(result["sufficient"])
        self.assertIn("证据不足", result["answer"])
        self.assertEqual(llm.answer_calls, 0)

    def test_missing_source_is_rejected(self) -> None:
        with self.assertRaises(RetrievalError):
            normalize_chunks({
                "status": "success",
                "data": {"chunks": [{"content": "text", "chunk_id": "chunk-001"}]},
                "metadata": {"query_mode": "naive"},
            })

    def test_insufficient_dense_evidence_advances_to_linear_and_preserves_pool(self) -> None:
        dense = ScriptedRetriever([{
            "evidence_id": "dense:d1", "content": "The 1925 Birthday Honours were made by George V.",
            "source_id": "d1", "source_ref": "dense-source", "retrieval_stage": "dense",
        }])
        linear = ScriptedRetriever([{
            "evidence_id": "linear:l1", "content": "George V was born in 1865.",
            "source_id": "l1", "source_ref": "linear-source", "retrieval_stage": "linear",
        }])
        llm = CascadeLLM()
        result = asyncio.run(build_agent_graph(dense, llm, linear).ainvoke({
            "query": "In which year was the King who made the 1925 Birthday Honours born?"
        }))
        self.assertEqual(result["current_stage"], "linear")
        self.assertEqual(dense.calls, 1)
        self.assertEqual(linear.calls, 1)
        self.assertEqual(llm.judge_calls, 2)
        self.assertEqual(result["evidence_pool"][0]["source_id"], "d1")
        self.assertEqual(result["current_evidence"][0]["source_id"], "l1")
        self.assertEqual(result["stage_evidence"]["dense"][0]["source_id"], "d1")
        self.assertEqual(result["stage_evidence"]["linear"][0]["source_id"], "l1")
        self.assertEqual(
            [event["reason"] for event in result["stage_history"] if event.get("node") == "judge"],
            ["缺少出生年份", "两条证据可以回答"],
        )
        self.assertEqual(result["answer_source_ids"], ["d1", "l1"])
        self.assertTrue(result["sufficient"])

    def test_sufficient_dense_evidence_skips_linear(self) -> None:
        linear = ScriptedRetriever([])
        result = asyncio.run(build_agent_graph(self.make_retriever(), FakeLLM(True), linear).ainvoke({
            "query": "who wrote it?"
        }))
        self.assertEqual(result["current_stage"], "dense")
        self.assertEqual(linear.calls, 0)

    def test_insufficient_linear_evidence_advances_to_local_with_pool(self) -> None:
        dense = ScriptedRetriever([{
            "evidence_id": "dense:d1", "content": "The 1925 Birthday Honours were made by George V.",
            "source_id": "d1", "source_ref": "dense-source", "retrieval_stage": "dense",
        }])
        linear = ScriptedRetriever([{
            "evidence_id": "linear:l1", "content": "The 1925 Birthday Honours are a list of British honours.",
            "source_id": "l1", "source_ref": "linear-source", "retrieval_stage": "linear",
        }])
        local = ScriptedRetriever([{
            "evidence_id": "light:g1", "content": "George V was born in 1865.",
            "source_id": "g1", "source_ref": "local-source", "retrieval_stage": "light",
        }])

        class ThirdStageLLM:
            def __init__(self):
                self.calls = 0

            async def judge(self, query, evidence):
                self.calls += 1
                return JudgeDecision(self.calls == 3, "测试判断", [], [1, 3] if self.calls == 3 else [])

            async def answer(self, query, evidence):
                return "1865。[1][3]", [evidence[0]["source_id"], evidence[-1]["source_id"]]

        llm = ThirdStageLLM()
        result = asyncio.run(build_agent_graph(dense, llm, linear, local).ainvoke({
            "query": "In which year was the King who made the 1925 Birthday Honours born?"
        }))
        self.assertEqual([dense.calls, linear.calls, local.calls], [1, 1, 1])
        self.assertEqual(llm.calls, 3)
        self.assertEqual(result["current_stage"], "light")
        self.assertEqual([item["source_id"] for item in result["evidence_pool"]], ["d1", "l1"])
        self.assertEqual(result["current_evidence"][0]["source_id"], "g1")
        self.assertEqual([e["stage"] for e in result["stage_history"] if e.get("tool")], ["dense", "linear", "light"])
        self.assertEqual(result["answer_source_ids"], ["g1", "l1"])
        self.assertEqual([item["source_id"] for item in result["answer_evidence"]], ["g1", "l1"])

    def test_sufficient_linear_evidence_skips_local(self) -> None:
        local = ScriptedRetriever([])
        linear = ScriptedRetriever([{
            "evidence_id": "linear:l1", "content": "The answer is 1865.",
            "source_id": "l1", "source_ref": "linear-source", "retrieval_stage": "linear",
        }])
        result = asyncio.run(build_agent_graph(self.make_retriever(), CascadeLLM(), linear, local).ainvoke({
            "query": "In which year was the King who made the 1925 Birthday Honours born?"
        }))
        self.assertEqual(result["current_stage"], "linear")
        self.assertEqual(local.calls, 0)

    def test_local_response_requires_local_mode_and_source(self) -> None:
        payload = {"status": "success", "metadata": {"query_mode": "local"}, "data": {
            "entities": [{"entity_name": "Joseph Glidden", "description": "Born January 18, 1813.",
                          "source_id": "chunk-source", "file_path": "hotpotqa/doc-1"}],
            "relationships": [],
            "chunks": [{"chunk_id": "c1", "content": "George V was born in 1865.", "file_path": "hotpotqa/doc-1"}]
        }}
        evidence = normalize_local_chunks(payload)
        self.assertEqual(evidence[0]["retrieval_stage"], "light")
        self.assertEqual(evidence[0]["source_id"], "entity:Joseph Glidden")
        self.assertEqual(evidence[0]["reference_id"], "chunk-source")
        self.assertEqual(evidence[1]["source_id"], "c1")
        with self.assertRaises(RetrievalError):
            normalize_local_chunks({**payload, "metadata": {"query_mode": "naive"}})
        with self.assertRaises(RetrievalError):
            normalize_local_chunks({**payload, "data": {"entities": [], "relationships": [],
                                                        "chunks": [{"content": "text", "chunk_id": "c1"}]}})

    def test_judge_context_keeps_long_passage_fact(self) -> None:
        passage = "background " * 300 + "Joseph Glidden was born January 18, 1813."
        context = evidence_context([{
            "evidence_id": "light:chunk:c1", "content": passage,
            "source_id": "c1", "source_ref": "hotpotqa/doc-1", "retrieval_stage": "light",
        }])
        self.assertIn("January 18, 1813", context)


if __name__ == "__main__":
    unittest.main()
