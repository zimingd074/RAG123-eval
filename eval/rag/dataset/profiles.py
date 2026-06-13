"""Evaluation profile selection and run metadata helpers."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from eval.common.schemas import EvalRecord, EvalSample

EvaluationProfile = Literal["static-v1", "tool-deferred", "all"]
PROFILE_CHOICES = ("static-v1", "tool-deferred", "all")
DEFAULT_PROFILE: EvaluationProfile = "static-v1"


def select_samples(
    samples: list[EvalSample],
    profile: EvaluationProfile,
) -> tuple[list[EvalSample], list[EvalSample]]:
    """Return samples included and excluded by an evaluation profile."""
    if profile == "all":
        return list(samples), []
    included = [s for s in samples if s.evaluation_scope == profile]
    excluded = [s for s in samples if s.evaluation_scope != profile]
    return included, excluded


def dataset_sha256(dataset_path: Path) -> str:
    """Return the SHA-256 digest of an evaluation dataset."""
    return hashlib.sha256(dataset_path.read_bytes()).hexdigest()


def metadata_path(runs_file: Path) -> Path:
    """Return the sidecar metadata path for a run JSONL file."""
    return runs_file.with_suffix(".meta.json")


def write_run_metadata(runs_file: Path, metadata: dict) -> Path:
    """Write run metadata without mixing non-record objects into JSONL."""
    path = metadata_path(runs_file)
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_run_metadata(runs_file: Path) -> dict:
    """Load run metadata; old runs without a sidecar return an empty dict."""
    path = metadata_path(runs_file)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_sample_annotations(
    record: EvalRecord,
    sample: EvalSample,
) -> EvalRecord:
    """Copy profile annotations from a dataset sample to an existing record."""
    payload = asdict(record)
    payload.update(
        {
            "expected_route": sample.expected_route,
            "evaluation_scope": sample.evaluation_scope,
            "scope_reason": sample.scope_reason,
            "annotation_rationale": sample.annotation_rationale,
            "requires_tool": sample.requires_tool,
        }
    )
    return EvalRecord.from_dict(payload)
