"""Tests for local environment configuration."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval.common.env import load_project_env


class ProjectEnvTest(unittest.TestCase):
    """Verify dotenv loading without overriding process configuration."""

    def test_loads_missing_values_and_preserves_existing_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "JUDGE_API_KEY=local-judge\n"
                "AIHUBMIX_API_KEY='local-embedding'\n"
                "# ignored\n",
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"JUDGE_API_KEY": "process-judge"},
                clear=True,
            ):
                load_project_env(env_path)

                self.assertEqual(os.environ["JUDGE_API_KEY"], "process-judge")
                self.assertEqual(
                    os.environ["AIHUBMIX_API_KEY"],
                    "local-embedding",
                )

    def test_missing_file_is_ignored(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            load_project_env(Path("missing.env"))

            self.assertNotIn("JUDGE_API_KEY", os.environ)


if __name__ == "__main__":
    unittest.main()
