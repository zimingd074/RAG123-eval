"""Tests for formal A/B comparability gates."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.rag.report.diff import _is_regression, compare


class ReportDiffTest(unittest.TestCase):
    """Only identical dataset hashes and profiles are comparable."""

    def write_scores(self, path: Path, digest: str, profile: str) -> None:
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "run_metadata": {
                        "dataset_sha256": digest,
                        "profile": profile,
                    },
                    "metrics": [],
                }
            ),
            encoding="utf-8",
        )

    def test_rejects_different_dataset_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a" / "_scores.json"
            second = Path(tmp) / "b" / "_scores.json"
            self.write_scores(first, "aaa", "static-v1")
            self.write_scores(second, "bbb", "static-v1")

            with self.assertRaisesRegex(ValueError, "哈希不同"):
                compare(str(first), str(second))

    def test_rejects_different_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a" / "_scores.json"
            second = Path(tmp) / "b" / "_scores.json"
            self.write_scores(first, "same", "static-v1")
            self.write_scores(second, "same", "all")

            with self.assertRaisesRegex(ValueError, "Profile 不同"):
                compare(str(first), str(second))

    def test_total_mean_latency_uses_lower_is_better_direction(self) -> None:
        self.assertFalse(_is_regression("total_mean_ms", -1241))
        self.assertTrue(_is_regression("total_mean_ms", 1001))
