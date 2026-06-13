"""Dataset selection tests for the RAG evaluation CLI."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from eval.common.cli import build_parser
from eval.common.schemas import load_samples
from eval.rag.dataset.annotate_profiles import DEFERRED_ROUTES
from eval.rag.dataset.profiles import select_samples


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "eval" / "rag" / "dataset"


class DatasetSelectionTest(unittest.TestCase):
    """Verify full dataset and profile selection behavior."""

    def test_default_dataset_is_full_with_static_profile(self) -> None:
        args = build_parser().parse_args(["rag", "run"])

        self.assertEqual(len(load_samples(args.dataset)), 150)
        self.assertEqual(args.profile, "static-v1")
        self.assertIsNone(args.limit)

    def test_full_dataset_has_one_hundred_fifty_samples(self) -> None:
        full_dataset = DATASET_DIR / "eval_set_v1_all.jsonl"
        args = build_parser().parse_args(
            ["rag", "run", "--dataset", str(full_dataset), "--limit", "150"]
        )

        self.assertEqual(args.dataset, full_dataset)
        self.assertEqual(len(load_samples(args.dataset)), 150)

    def test_profiles_partition_full_dataset(self) -> None:
        samples = load_samples(DATASET_DIR / "eval_set_v1_all.jsonl")
        static, static_excluded = select_samples(samples, "static-v1")
        deferred, deferred_excluded = select_samples(samples, "tool-deferred")

        self.assertEqual(len(static), 127)
        self.assertEqual(len(static_excluded), 23)
        self.assertEqual(len(deferred), 23)
        self.assertEqual(len(deferred_excluded), 127)
        self.assertEqual(
            {sample.query_id for sample in deferred},
            set(DEFERRED_ROUTES),
        )

    def test_smoke_dataset_remains_twenty_samples(self) -> None:
        smoke = DATASET_DIR / "eval_set_v1.jsonl"

        self.assertEqual(len(load_samples(smoke)), 20)

    def test_smoke_ground_truth_matches_full_dataset(self) -> None:
        full_samples = {
            sample.query_id: sample
            for sample in load_samples(DATASET_DIR / "eval_set_v1_all.jsonl")
        }
        smoke_samples = load_samples(DATASET_DIR / "eval_set_v1.jsonl")

        for sample in smoke_samples:
            self.assertEqual(
                sample.ground_truth,
                full_samples[sample.query_id].ground_truth,
                sample.query_id,
            )

    def test_annotations_and_document_ids_are_consistent(self) -> None:
        samples = load_samples(DATASET_DIR / "eval_set_v1_all.jsonl")
        known_doc_ids: set[str] = set()
        for path in (PROJECT_ROOT / "knowledge_base").rglob("*.md"):
            match = re.search(
                r'^doc_id:\s*["\']?([^"\'\s]+)',
                path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
            if match:
                known_doc_ids.add(match.group(1))

        for sample in samples:
            self.assertEqual(
                sample.requires_tool,
                sample.expected_route in {"TOOL", "HYBRID"},
            )
            self.assertTrue(sample.scope_reason)
            self.assertTrue(sample.annotation_rationale)
            for doc_id in sample.expected_doc_ids + sample.expected_doc_ids_nice:
                self.assertNotEqual(doc_id, "PRODUCT_MAPPING")
                self.assertIn(doc_id, known_doc_ids)


if __name__ == "__main__":
    unittest.main()
