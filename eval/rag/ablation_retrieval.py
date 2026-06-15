"""Run retrieval-only Weak-20 ablations against ragent /rag/eval."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from eval.common.env import load_project_env
from eval.common.schemas import load_samples
from eval.rag.pipeline.runner import fetch_eval_retrieval, fetch_trace_detail, login

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    PROJECT_ROOT
    / "eval"
    / "rag"
    / "dataset"
    / "eval_set_static_weak20_20260613_groundtruth_fixed.jsonl"
)


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * quantile))]


def _trace_metrics(detail: dict[str, Any] | None) -> dict[str, Any]:
    nodes = list((detail or {}).get("nodes") or [])
    retrieval_ms = [
        float(node.get("durationMs") or 0)
        for node in nodes
        if node.get("nodeName") == "multi-channel-retrieval"
    ]
    keyword_nodes = [
        node for node in nodes if node.get("nodeName") == "keyword-pg-search"
    ]
    return {
        "retrieval_ms": retrieval_ms,
        "keyword_ms": [
            float(node.get("durationMs") or 0) for node in keyword_nodes
        ],
        "keyword_candidates": [
            float((node.get("extraData") or {}).get("candidateCount") or 0)
            for node in keyword_nodes
        ],
        "ordinary_fts_enabled": [
            bool((node.get("extraData") or {}).get("ordinaryFtsEnabled"))
            for node in keyword_nodes
        ],
    }


def _sample_metrics(expected: list[str], retrieved: list[str]) -> dict[str, float]:
    expected_set = set(expected)
    top5 = retrieved[:5]
    hit5 = float(bool(expected_set.intersection(top5)))
    recall5 = (
        len(expected_set.intersection(top5)) / len(expected_set)
        if expected_set
        else 0.0
    )
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved[:10], start=1):
        if doc_id in expected_set:
            reciprocal_rank = 1.0 / rank
            break
    return {"hit5": hit5, "recall5": recall5, "mrr10": reciprocal_rank}


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_project_env()
    base_url = os.environ.get(
        "RAGENT_BASE_URL", "http://localhost:9090/api/ragent"
    ).rstrip("/")
    username = os.environ.get("RAGENT_USERNAME", "admin")
    password = os.environ.get("RAGENT_PASSWORD", "admin")
    token = login(base_url, username, password)
    samples = load_samples(Path(args.dataset))

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        started = time.perf_counter()
        state = fetch_eval_retrieval(
            base_url,
            token,
            sample.query,
            intent_leaf_id=sample.intent_l2,
        )
        wall_ms = (time.perf_counter() - started) * 1000
        trace = fetch_trace_detail(
            base_url,
            token,
            trace_id=state.get("trace_id"),
            retries=10,
        )
        retrieved = list(state.get("retrieved_doc_ids_ragent") or [])
        metrics = _sample_metrics(sample.expected_doc_ids, retrieved)
        row = {
            "query_id": sample.query_id,
            "query": sample.query,
            "expected_doc_ids": sample.expected_doc_ids,
            "retrieved_doc_ids": retrieved,
            "trace_id": state.get("trace_id"),
            "intent_leaf_ids": state.get("intent_leaf_ids"),
            "error": state.get("error"),
            "wall_ms": wall_ms,
            **metrics,
            **_trace_metrics(trace),
        }
        rows.append(row)
        print(
            f"[{index:>2}/{len(samples)}] {sample.query_id:<7} "
            f"hit5={int(metrics['hit5'])} "
            f"recall5={metrics['recall5']:.3f} "
            f"wall={wall_ms:.0f}ms"
        )
        if args.request_delay_ms > 0 and index < len(samples):
            time.sleep(args.request_delay_ms / 1000)

    retrieval_calls = [
        value for row in rows for value in row["retrieval_ms"]
    ]
    keyword_calls = [value for row in rows for value in row["keyword_ms"]]
    keyword_candidates = [
        value for row in rows for value in row["keyword_candidates"]
    ]
    summary = {
        "label": args.label,
        "rrf_k": args.rrf_k,
        "keyword_multiplier": args.keyword_multiplier,
        "ordinary_fts_conditional": args.ordinary_fts_conditional,
        "sample_count": len(rows),
        "error_count": sum(bool(row["error"]) for row in rows),
        "hit5": statistics.fmean(row["hit5"] for row in rows),
        "recall5": statistics.fmean(row["recall5"] for row in rows),
        "mrr10": statistics.fmean(row["mrr10"] for row in rows),
        "wall_p50_ms": _percentile([row["wall_ms"] for row in rows], 0.50),
        "wall_p95_ms": _percentile([row["wall_ms"] for row in rows], 0.95),
        "retrieval_p50_ms": _percentile(retrieval_calls, 0.50),
        "retrieval_p95_ms": _percentile(retrieval_calls, 0.95),
        "keyword_p95_ms": _percentile(keyword_calls, 0.95),
        "keyword_candidate_mean": (
            statistics.fmean(keyword_candidates) if keyword_candidates else 0.0
        ),
        "keyword_candidate_max": max(keyword_candidates, default=0.0),
        "ordinary_fts_call_count": sum(
            enabled
            for row in rows
            for enabled in row["ordinary_fts_enabled"]
        ),
    }
    result = {"summary": summary, "samples": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={output.resolve()}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--rrf-k", type=int, required=True)
    parser.add_argument("--keyword-multiplier", type=int, required=True)
    parser.add_argument("--request-delay-ms", type=int, default=0)
    parser.add_argument(
        "--ordinary-fts-conditional",
        action=argparse.BooleanOptionalAction,
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
