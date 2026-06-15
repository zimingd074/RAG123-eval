from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from eval.common.cli import build_parser
from eval.common.schemas import load_samples
from eval.rag.chat_model_benchmark import (
    cost_per_1000,
    final_rank,
    routing_metrics,
    write_pareto_svg,
)
from eval.rag.pipeline.runner import (
    _parse_extra_data,
    retrieval_state_from_chat_trace,
    stream_chat_one_query,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = (
    PROJECT_ROOT
    / "eval"
    / "rag"
    / "dataset"
    / "eval_set_chat_models20_20260615.jsonl"
)


class ChatModelBenchmarkTest(unittest.TestCase):
    def test_fixed_dataset_distribution_and_intent_cap(self) -> None:
        samples = load_samples(DATASET)
        difficulties = {
            difficulty: sum(sample.difficulty == difficulty for sample in samples)
            for difficulty in ("hard", "medium", "easy")
        }
        intent_counts: dict[str, int] = {}
        for sample in samples:
            intent_counts[sample.intent_l2] = intent_counts.get(sample.intent_l2, 0) + 1

        self.assertEqual(len(samples), 20)
        self.assertEqual(difficulties, {"hard": 6, "medium": 8, "easy": 6})
        self.assertLessEqual(max(intent_counts.values()), 2)
        self.assertTrue(all(sample.expected_route == "KB" for sample in samples))
        self.assertTrue(all(not sample.requires_tool for sample in samples))

    def test_cli_exposes_three_round_final(self) -> None:
        args = build_parser().parse_args(["rag", "chat-model-benchmark"])

        self.assertEqual(args.top_routing, 2)
        self.assertEqual(args.finalists, 3)
        self.assertEqual(args.final_rounds, 3)

    def test_stream_chat_sends_request_scoped_models(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.iter_content.return_value = [
            b"event: finish\ndata: {}\n\nevent: done\ndata: [DONE]\n\n"
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)

        with patch("eval.rag.pipeline.runner.requests.get", return_value=response) as get:
            state = stream_chat_one_query(
                "http://localhost",
                "token",
                "question",
                routing_model="qwen3.6-flash",
                answer_model="qwen3.7-plus",
            )

        self.assertEqual(state["final_status"], "success")
        self.assertEqual(
            get.call_args.kwargs["params"],
            {
                "question": "question",
                "routingModelId": "qwen3.6-flash",
                "answerModelId": "qwen3.7-plus",
            },
        )

    def test_same_trace_reconstructs_retrieval_and_intent(self) -> None:
        trace = {
            "run": {"traceId": "trace-1"},
            "nodes": [
                {
                    "nodeName": "eval-intent-result",
                    "extraData": {
                        "subIntents": [
                            {"candidates": [{"intentId": "S1", "score": 0.9}]}
                        ]
                    },
                },
                {
                    "nodeName": "eval-retrieval-result",
                    "extraData": json.dumps(
                        {
                            "retrievedDocIds": ["D1"],
                            "retrievedChunkIds": ["C1"],
                            "retrievedContexts": ["context"],
                            "retrievedContextDocIds": ["D1"],
                            "hasKb": True,
                            "hasMcp": False,
                        }
                    ),
                },
            ],
        }
        trace["nodes"] = [
            {**node, "extraData": _parse_extra_data(node["extraData"])}
            for node in trace["nodes"]
        ]

        state = retrieval_state_from_chat_trace(trace)

        self.assertIsNotNone(state)
        self.assertEqual(state["trace_id"], "trace-1")
        self.assertEqual(state["retrieved_doc_ids_ragent"], ["D1"])
        self.assertEqual(state["intent_leaf_ids"], ["S1"])

    def test_complete_cost_includes_routing_and_answer(self) -> None:
        routing = {"mean_input_tokens": 1000, "mean_output_tokens": 100}
        answer = {"mean_input_tokens": 2000, "mean_output_tokens": 500}
        routing_price = {"input_per_million": 1, "output_per_million": 2}
        answer_price = {"input_per_million": 3, "output_per_million": 4}

        self.assertAlmostEqual(
            cost_per_1000(routing, answer, routing_price, answer_price),
            9.2,
        )

    def test_routing_metrics_use_manual_rewrite_review(self) -> None:
        record = SimpleNamespace(
            query_id="Q1",
            intent_pred="S1",
            intent_l2="S1",
            reference_doc_ids=["D1"],
            retrieved_doc_ids=["D1"],
            final_status="success",
            model_calls=[
                {
                    "step": "query-rewrite",
                    "parsed": True,
                    "fallback": False,
                    "estimatedInputTokens": 10,
                    "estimatedOutputTokens": 5,
                    "latencyMs": 20,
                }
            ],
        )

        metrics = routing_metrics(
            [record],
            {
                "Q1": {
                    "semantic_score": 0.5,
                    "silent_condition_change": False,
                }
            },
        )

        self.assertEqual(metrics["rewrite_semantic_score"], 0.5)
        self.assertTrue(metrics["human_review_complete"])
        self.assertEqual(metrics["mean_input_tokens"], 10)
        self.assertEqual(metrics["mean_latency_ms"], 20)

    def test_one_point_tie_prefers_lower_cost_then_ttft(self) -> None:
        baseline = {
            "answer": {
                "faithfulness": 0.9,
                "hard_answer_correctness": 0.8,
            },
            "routing": {"intent_accuracy": 0.9},
        }

        def arm(name: str, quality: float, cost: float, ttft: float) -> dict:
            return {
                "routing_model": name,
                "answer_model": name,
                "routing": {"intent_accuracy": 0.9, "quality": 0.9},
                "answer": {
                    "faithfulness": quality,
                    "hard_answer_correctness": 0.8,
                    "success_rate": 1.0,
                    "mean_ttft_ms": ttft,
                },
                "answer_quality": quality,
                "cost_per_1000": cost,
            }

        ranked = final_rank(
            [
                arm("higher-score", 0.905, 10.0, 100),
                arm("lower-cost", 0.900, 5.0, 120),
            ],
            baseline,
        )

        self.assertEqual(ranked[0]["routing_model"], "lower-cost")

    def test_pareto_svg_is_generated(self) -> None:
        arms = [
            {
                "routing_model": "routing",
                "answer_model": "answer",
                "cost_per_1000": 2.0,
                "answer_quality": 0.8,
                "eligible": True,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pareto.svg"
            write_pareto_svg(path, arms)

            self.assertIn("routing + answer", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
