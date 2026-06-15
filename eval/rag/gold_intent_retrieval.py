from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from eval.common.schemas import load_samples
from eval.rag.pipeline.runner import fetch_eval_retrieval, login


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(quantile * len(ordered) + 0.999999) - 1))
    return round(ordered[index], 1)


def run(dataset: Path, state_dir: Path) -> Path:
    base_url = os.environ.get("RAGENT_BASE_URL", "http://localhost:9090/api/ragent").rstrip("/")
    username = os.environ.get("RAGENT_USERNAME")
    password = os.environ.get("RAGENT_PASSWORD")
    if not username or not password:
        raise RuntimeError("RAGENT_USERNAME and RAGENT_PASSWORD are required")

    doc_map = json.loads((state_dir / "doc_id_map.json").read_text(encoding="utf-8"))
    ragent_to_biz = {value["ragent_doc_id"]: key for key, value in doc_map.items()}
    samples = load_samples(dataset)
    token = login(base_url, username, password)

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        # EvalController names this parameter intentLeafId, but the runtime
        # IntentNode id is the persisted intent_code, not the database row id.
        intent_id = sample.intent_l2
        started = time.perf_counter()
        result = fetch_eval_retrieval(base_url, token, sample.query, intent_id)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        retrieved = [
            ragent_to_biz.get(doc_id, doc_id)
            for doc_id in result["retrieved_doc_ids_ragent"]
        ]
        expected = list(sample.expected_doc_ids)
        rows.append(
            {
                "query_id": sample.query_id,
                "query": sample.query,
                "intent_l2": sample.intent_l2,
                "intent_leaf_id": intent_id,
                "expected_doc_ids": expected,
                "retrieved_doc_ids": retrieved,
                "latency_ms": latency_ms,
                "error": result["error"],
            }
        )
        print(
            f"[{index:>2}/{len(samples)}] {sample.query_id:<7} "
            f"docs={len(retrieved):<2} latency={latency_ms:>7.1f}ms"
        )

    metrics: dict[str, Any] = {}
    for k in (1, 3, 5, 10):
        hits = []
        recalls = []
        for row in rows:
            expected = set(row["expected_doc_ids"])
            top_k = row["retrieved_doc_ids"][:k]
            matched = expected.intersection(top_k)
            hits.append(bool(matched))
            recalls.append(len(matched) / len(expected) if expected else 1.0)
        metrics[f"hit@{k}"] = round(sum(hits) / len(hits), 4)
        metrics[f"recall@{k}"] = round(sum(recalls) / len(recalls), 4)

    reciprocal_ranks = []
    for row in rows:
        expected = set(row["expected_doc_ids"])
        rank = next(
            (index for index, doc_id in enumerate(row["retrieved_doc_ids"][:10], start=1)
             if doc_id in expected),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    latencies = [row["latency_ms"] for row in rows]
    metrics.update(
        {
            "mrr@10": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "error_rate": round(sum(bool(row["error"]) for row in rows) / len(rows), 4),
            "sample_count": len(rows),
        }
    )

    out_dir = state_dir / "gold_intent"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gold_intent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "dataset": str(dataset.resolve()),
        "state_dir": str(state_dir.resolve()),
        "base_url": base_url,
        "metrics": metrics,
        "rows": rows,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Output: {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval with the gold intent leaf fixed")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.state_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
