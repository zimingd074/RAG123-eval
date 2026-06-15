"""Build the fixed 20-query dataset used by the dual-chat-model benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "eval" / "rag" / "dataset"
DEFAULT_SOURCE = DATASET_DIR / "eval_set_v1_all.jsonl"
DEFAULT_OUTPUT = DATASET_DIR / "eval_set_chat_models20_20260615.jsonl"
TARGETS = {"hard": 6, "medium": 8, "easy": 6}
SEED = 20260615


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_rank(sample: dict) -> str:
    value = f"{SEED}:{sample['query_id']}".encode()
    return hashlib.sha256(value).hexdigest()


def select(samples: list[dict]) -> list[dict]:
    eligible = [
        sample
        for sample in samples
        if sample.get("evaluation_scope") == "static-v1"
        and sample.get("expected_route") == "KB"
        and not sample.get("requires_tool", False)
        and sample.get("ground_truth")
        and sample.get("expected_doc_ids")
    ]
    selected: list[dict] = []
    intent_counts: Counter[str] = Counter()

    for difficulty, target in TARGETS.items():
        pool = sorted(
            (sample for sample in eligible if sample.get("difficulty") == difficulty),
            key=stable_rank,
        )
        chosen: list[dict] = []
        while len(chosen) < target:
            candidates = [
                sample
                for sample in pool
                if sample not in chosen
                and intent_counts[sample["intent_l2"]] < 2
            ]
            if not candidates:
                raise RuntimeError(f"cannot satisfy target for {difficulty}")
            candidates.sort(
                key=lambda sample: (
                    intent_counts[sample["intent_l2"]],
                    sum(
                        item["intent_l2"] == sample["intent_l2"]
                        for item in chosen
                    ),
                    stable_rank(sample),
                )
            )
            item = candidates[0]
            chosen.append(item)
            intent_counts[item["intent_l2"]] += 1
        selected.extend(chosen)

    if Counter(sample["difficulty"] for sample in selected) != Counter(TARGETS):
        raise RuntimeError("difficulty distribution mismatch")
    if max(intent_counts.values(), default=0) > 2:
        raise RuntimeError("an intent appears more than twice")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    selected = select(load_jsonl(args.source))
    args.output.write_text(
        "".join(
            json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n"
            for sample in selected
        ),
        encoding="utf-8",
    )
    manifest = {
        "source": str(args.source.resolve()),
        "seed": SEED,
        "sample_count": len(selected),
        "difficulty_distribution": dict(
            Counter(sample["difficulty"] for sample in selected)
        ),
        "intent_distribution": dict(Counter(sample["intent_l2"] for sample in selected)),
        "query_ids": [sample["query_id"] for sample in selected],
        "constraints": {
            "evaluation_scope": "static-v1",
            "expected_route": "KB",
            "requires_tool": False,
            "max_samples_per_intent": 2,
        },
        "human_review_status": "pending",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
