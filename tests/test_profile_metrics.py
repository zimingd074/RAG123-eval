"""Tests for profile-aware schemas and metrics."""
from __future__ import annotations

import unittest

from eval.common.schemas import EvalRecord, EvalSample
from eval.rag.metrics import behavior, retrieval
from eval.rag.metrics.ragas_judge import (
    _strip_frontmatter,
    filter_evaluable,
)


def make_record(**overrides: object) -> EvalRecord:
    """Build a minimal successful record for metric tests."""
    values: dict[str, object] = {
        "query_id": "Q-1",
        "user_input": "question",
        "reference": "answer",
        "reference_doc_ids": ["DOC_MUST"],
        "reference_doc_ids_nice": ["DOC_NICE"],
        "intent_l1": "SUPPORT",
        "intent_l2": "S2_参数咨询",
        "difficulty": "easy",
        "requires_rag": True,
        "response": "answer",
        "thinking": None,
        "latency_ms": 100,
        "first_token_ms": 50,
        "final_status": "success",
        "error": None,
        "conversation_id": None,
        "task_id": None,
        "retrieved_doc_ids": ["DOC_NICE", "DOC_MUST"],
        "retrieved_doc_ids_raw": [],
        "retrieved_chunk_ids": [],
        "retrieved_contexts": ["context"],
        "retrieved_context_doc_ids": ["DOC_NICE"],
        "intent_pred": "S2_参数咨询",
        "intent_pred_all": ["S2_参数咨询"],
        "has_kb": True,
        "has_mcp": False,
        "trace_id": None,
        "expected_route": "KB",
        "evaluation_scope": "static-v1",
        "scope_reason": "",
        "annotation_rationale": "",
        "requires_tool": False,
    }
    values.update(overrides)
    return EvalRecord(**values)


class ProfileSchemaTest(unittest.TestCase):
    """Verify old records get route defaults without losing compatibility."""

    def test_legacy_sample_route_follows_requires_rag(self) -> None:
        sample = EvalSample.from_dict(
            {
                "query_id": "Q",
                "query": "hello",
                "intent_l1": "CHAT",
                "intent_l2": "C1_寒暄问候",
                "difficulty": "easy",
                "requires_rag": False,
                "expected_doc_ids": [],
            }
        )

        self.assertEqual(sample.expected_route, "SYSTEM")
        self.assertEqual(sample.evaluation_scope, "static-v1")


class ProfileMetricTest(unittest.TestCase):
    """Verify route-aware metric eligibility and renamed recall semantics."""

    def test_retrieval_metrics_separate_must_and_nice(self) -> None:
        metrics = {metric.name: metric for metric in retrieval.compute([make_record()])}

        self.assertEqual(metrics["hit@1"].overall, 0.0)
        self.assertEqual(metrics["recall@1"].overall, 0.0)
        self.assertEqual(metrics["recall_all_expected@1"].overall, 0.5)
        self.assertEqual(metrics["nice_hit@1"].overall, 1.0)
        self.assertNotIn("recall_inclusive@1", metrics)

    def test_system_behavior_requires_no_retrieval(self) -> None:
        clean = make_record(
            query_id="SYS-1",
            requires_rag=False,
            expected_route="SYSTEM",
            reference_doc_ids=[],
            reference_doc_ids_nice=[],
            retrieved_doc_ids=[],
            retrieved_contexts=[],
            has_kb=False,
        )
        over = make_record(
            query_id="SYS-2",
            requires_rag=False,
            expected_route="SYSTEM",
            reference_doc_ids=[],
            reference_doc_ids_nice=[],
            retrieved_doc_ids=["DOC"],
        )
        metrics = {metric.name: metric for metric in behavior.compute([clean, over])}

        self.assertEqual(metrics["over_retrieval_rate"].overall, 0.5)
        self.assertEqual(metrics["system_boundary_compliance"].overall, 0.5)

    def test_tool_deferred_is_excluded_from_core_metrics(self) -> None:
        deferred = make_record(
            evaluation_scope="tool-deferred",
            expected_route="TOOL",
            requires_tool=True,
        )
        metrics = {metric.name: metric for metric in retrieval.compute([deferred])}

        self.assertIsNone(metrics["hit@5"].overall)
        self.assertIsNone(metrics["hit@5"].per_sample["Q-1"])

    def test_ragas_filters_use_semantic_skip_reasons(self) -> None:
        system = make_record(
            query_id="SYS",
            expected_route="SYSTEM",
            requires_rag=False,
            reference_doc_ids=[],
            reference_doc_ids_nice=[],
            retrieved_doc_ids=[],
            retrieved_contexts=[],
        )
        deferred = make_record(
            query_id="TOOL",
            expected_route="TOOL",
            evaluation_scope="tool-deferred",
            requires_tool=True,
        )

        _, kb_skipped = filter_evaluable([system, deferred])

        self.assertEqual(
            dict(kb_skipped),
            {"SYS": "expected_system", "TOOL": "tool_deferred"},
        )

    def test_frontmatter_can_be_removed_for_ab_scoring(self) -> None:
        content = "---\ndoc_id: PROD_1\ntitle: Demo\n---\n正文"

        self.assertEqual(_strip_frontmatter(content), "正文")
