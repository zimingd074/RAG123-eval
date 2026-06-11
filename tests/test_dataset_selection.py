"""Dataset selection tests for the RAG evaluation CLI."""
from __future__ import annotations

import unittest
from pathlib import Path

from eval.common.cli import build_parser
from eval.common.schemas import load_samples


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "eval" / "rag" / "dataset"


class DatasetSelectionTest(unittest.TestCase):
    """Verify the CLI can select the full evaluation dataset."""

    def test_default_dataset_has_twenty_samples(self) -> None:
        args = build_parser().parse_args(["rag", "run"])

        self.assertEqual(len(load_samples(args.dataset)), 20)

    def test_full_dataset_has_one_hundred_fifty_samples(self) -> None:
        full_dataset = DATASET_DIR / "eval_set_v1_all.jsonl"
        args = build_parser().parse_args(
            ["rag", "run", "--dataset", str(full_dataset), "--limit", "150"]
        )

        self.assertEqual(args.dataset, full_dataset)
        self.assertEqual(len(load_samples(args.dataset)), 150)


if __name__ == "__main__":
    unittest.main()
