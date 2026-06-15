from __future__ import annotations

import unittest
from types import SimpleNamespace

from eval.rag.metrics import rewrite


def make_record(query_id: str, original: str, rewritten: str) -> SimpleNamespace:
    return SimpleNamespace(
        query_id=query_id,
        user_input=original,
        intent_l1="SUPPORT",
        intent_l2="S1",
        evaluation_scope="static-v1",
        chat_trace={
            "nodes": [
                {
                    "nodeName": "eval-query-rewrite",
                    "extraData": {
                        "originalQuestion": original,
                        "rewrittenQuestion": rewritten,
                    },
                }
            ]
        },
    )


class RewriteMetricsTest(unittest.TestCase):
    def test_equivalent_time_and_negation_are_preserved(self) -> None:
        evaluation = rewrite.evaluate_record(
            make_record("Q1", "我下单了 3 天还没发货", "下单3天未发货")
        )

        self.assertFalse(evaluation.silent_condition_change)
        self.assertEqual(evaluation.condition_recall, 1.0)
        self.assertGreater(evaluation.semantic_score or 0, 0.8)

    def test_changed_quantity_is_a_silent_condition_change(self) -> None:
        evaluation = rewrite.evaluate_record(
            make_record("Q2", "我下单了 3 天还没发货", "下单7天未发货")
        )

        self.assertTrue(evaluation.silent_condition_change)
        self.assertIn("quantity:3天", evaluation.missing_conditions)
        self.assertIn("quantity:7天", evaluation.added_conditions)

    def test_dropped_question_target_is_detected(self) -> None:
        evaluation = rewrite.evaluate_record(
            make_record(
                "Q3",
                "小米智能门锁 E10 支持几种开锁方式？",
                "小米智能门锁 E10 支持的开锁方式",
            )
        )

        self.assertTrue(evaluation.silent_condition_change)
        self.assertIn("question:count", evaluation.missing_conditions)

    def test_keyword_style_rewrite_does_not_require_question_word(self) -> None:
        evaluation = rewrite.evaluate_record(
            make_record(
                "Q5",
                "扫地机的水箱可以用洗洁精洗吗？",
                "扫地机水箱可以用洗洁精清洗",
            )
        )

        self.assertFalse(evaluation.silent_condition_change)

    def test_model_change_is_detected(self) -> None:
        evaluation = rewrite.evaluate_record(
            make_record("Q4", "Redmi K70 支持无线充电吗？", "Redmi K80 支持无线充电吗")
        )

        self.assertTrue(evaluation.silent_condition_change)
        self.assertIn("entity:redmik70", evaluation.missing_conditions)
        self.assertIn("entity:redmik80", evaluation.added_conditions)

    def test_compute_exposes_formal_metrics_and_change_details(self) -> None:
        records = [
            make_record("Q1", "退货运费谁出？", "退货运费由谁承担"),
            make_record("Q2", "买了 3 天能退吗？", "买了7天能退吗"),
        ]

        metrics = {metric.name: metric for metric in rewrite.compute(records)}

        self.assertEqual(
            set(metrics),
            {
                "rewrite_parse_rate",
                "rewrite_semantic_preservation",
                "rewrite_condition_recall",
                "silent_condition_change_rate",
            },
        )
        self.assertEqual(metrics["rewrite_parse_rate"].overall, 1.0)
        self.assertEqual(metrics["silent_condition_change_rate"].overall, 0.5)
        self.assertIn(
            "Q2",
            metrics["silent_condition_change_rate"].meta["changes"],
        )


if __name__ == "__main__":
    unittest.main()
