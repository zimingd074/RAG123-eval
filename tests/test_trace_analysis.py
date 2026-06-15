"""Tests for retrieval trace report aggregation."""
from __future__ import annotations

import unittest

from eval.rag.report.trace_analysis import analyze
from tests.test_profile_metrics import make_record


class TraceAnalysisTest(unittest.TestCase):
    """Verify stage percentiles and rerank degradation reporting."""

    def test_aggregates_stage_latency_and_bottleneck(self) -> None:
        record = make_record(
            chat_trace={
                "nodes": [
                    {
                        "nodeName": "multi-channel-retrieval",
                        "durationMs": 2600,
                        "extraData": {},
                    },
                    {
                        "nodeName": "vector-global-search",
                        "durationMs": 300,
                        "extraData": {"candidateCount": 15},
                    },
                    {
                        "nodeName": "rerank",
                        "durationMs": 2300,
                        "extraData": {
                            "inputCandidates": 15,
                            "fallbackToRrf": True,
                            "timedOut": True,
                        },
                    },
                ]
            }
        )

        summary = analyze([record])

        self.assertEqual(summary["stages"][0]["stage"], "vector-global-search")
        self.assertEqual(summary["rerank_fallback_rate"], 1.0)
        self.assertEqual(summary["rerank_timeout_rate"], 1.0)
        self.assertEqual(summary["retrieval_latency"]["p95_ms"], 2600)
        self.assertEqual(
            summary["bottlenecks"][0]["bottleneck_stage"],
            "rerank",
        )

    def test_falls_back_to_eval_trace_when_chat_trace_has_no_retrieval(self) -> None:
        record = make_record(
            chat_trace={
                "nodes": [
                    {
                        "nodeName": "user-first-packet",
                        "durationMs": 1800,
                        "extraData": {},
                    }
                ]
            },
            eval_trace={
                "nodes": [
                    {
                        "nodeName": "vector-global-search",
                        "durationMs": 120,
                        "extraData": {"candidateCount": 10},
                    },
                    {
                        "nodeName": "rerank",
                        "durationMs": 200,
                        "extraData": {
                            "inputCandidates": 10,
                            "fallbackToRrf": False,
                        },
                    },
                ]
            },
        )

        summary = analyze([record])

        self.assertEqual(summary["stages"][0]["count"], 1)
        self.assertEqual(summary["rerank_count"], 1)
        self.assertEqual(summary["slowest"]["node"], "user-first-packet")

    def test_stage_counts_track_multiple_retrieval_invocations(self) -> None:
        record = make_record(
            chat_trace={
                "nodes": [
                    {
                        "nodeName": "rerank",
                        "durationMs": 100,
                        "extraData": {"inputCandidates": 10},
                    },
                    {
                        "nodeName": "rerank",
                        "durationMs": 200,
                        "extraData": {"inputCandidates": 15},
                    },
                ]
            }
        )

        summary = analyze([record])

        self.assertEqual(summary["stages"][0]["count"], 2)
        self.assertEqual(summary["stages"][0]["candidate_mean"], 12.5)
        self.assertEqual(summary["rerank_count"], 2)


if __name__ == "__main__":
    unittest.main()
