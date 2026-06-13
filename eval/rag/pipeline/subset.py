"""Derive a profile-specific run from an existing recorded run."""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from pathlib import Path

from eval.common.schemas import load_samples
from eval.rag.dataset.profiles import (
    DEFAULT_PROFILE,
    EvaluationProfile,
    apply_sample_annotations,
    dataset_sha256,
    load_run_metadata,
    select_samples,
    write_run_metadata,
)
from eval.rag.pipeline.runner import EVAL_SET_PATH, RUNS_DIR
from eval.rag.pipeline.score import load_records


def derive_profile_run(
    source_runs_file: Path,
    *,
    dataset_path: Path = EVAL_SET_PATH,
    profile: EvaluationProfile = DEFAULT_PROFILE,
    out_path: Path | None = None,
) -> Path:
    """Create a filtered run using annotations from the specified dataset.

    This is the only supported way to apply new profile annotations to an old
    recording. It produces a new run and keeps the source run immutable.
    """
    records = load_records(source_runs_file)
    samples = load_samples(dataset_path)
    included, excluded = select_samples(samples, profile)
    included_by_id = {sample.query_id: sample for sample in included}
    source_by_id = {record.query_id: record for record in records}
    missing = sorted(set(included_by_id) - set(source_by_id))
    if missing:
        raise RuntimeError(
            "源 run 不覆盖所选 Profile，缺少样本："
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    derived = [
        apply_sample_annotations(source_by_id[sample.query_id], sample)
        for sample in included
    ]
    RUNS_DIR.mkdir(exist_ok=True)
    out_path = out_path or (
        RUNS_DIR
        / f"v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{profile}.jsonl"
    )
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in derived:
            handle.write(
                json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n"
            )

    source_meta = load_run_metadata(source_runs_file)
    write_run_metadata(
        out_path,
        {
            "dataset_path": str(dataset_path.resolve()),
            "dataset_sha256": dataset_sha256(dataset_path),
            "profile": profile,
            "original_sample_count": len(samples),
            "profile_sample_count": len(included),
            "selected_sample_count": len(derived),
            "excluded_sample_count": len(excluded),
            "excluded_sample_ids": [sample.query_id for sample in excluded],
            "excluded_samples": [
                {"query_id": sample.query_id, "intent_l2": sample.intent_l2}
                for sample in excluded
            ],
            "source_runs_file": str(source_runs_file.resolve()),
            "source_run_metadata": source_meta,
            "derived_from_recording": True,
        },
    )
    return out_path
