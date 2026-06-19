"""Run local PostgreSQL keyword-only evaluation.

This script intentionally does not call ragent or any model provider. It queries the
local postgres Docker container directly and maps ragent doc IDs back to business
IDs via eval/rag/dataset/doc_id_map.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from eval.common.schemas import load_samples

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOC_MAP = PROJECT_ROOT / "eval" / "rag" / "dataset" / "doc_id_map.json"


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * quantile))]


def _load_doc_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {value["ragent_doc_id"]: key for key, value in payload.items()}


def _dollar_quote(value: str) -> str:
    tag = "bm25q"
    while f"${tag}$" in value:
        tag += "x"
    return f"${tag}${value}${tag}$"


def _query_bm25(container: str, database: str, user: str, query: str, top_k: int) -> list[dict[str, Any]]:
    sql = f"""
WITH hits AS (
    SELECT metadata->>'doc_id' AS doc_id,
           max(pdb.score(id)) AS score
      FROM t_knowledge_vector
     WHERE content ||| {_dollar_quote(query)}
     GROUP BY metadata->>'doc_id'
     ORDER BY max(pdb.score(id)) DESC, metadata->>'doc_id'
     LIMIT {top_k}
)
SELECT coalesce(json_agg(json_build_object('doc_id', doc_id, 'score', score)), '[]'::json)
  FROM hits;
"""
    completed = subprocess.run(
        ["docker", "exec", "-i", container, "psql", "-U", user, "-d", database, "-At", "-c", sql],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    output = completed.stdout.strip()
    return json.loads(output or "[]")


def _query_fts(container: str, database: str, user: str, query: str, top_k: int) -> list[dict[str, Any]]:
    sql = f"""
WITH q AS (
    SELECT websearch_to_tsquery('simple', {_dollar_quote(query)}) AS query
),
hits AS (
    SELECT metadata->>'doc_id' AS doc_id,
           max(ts_rank_cd(search_vector, q.query)) AS score
      FROM t_knowledge_vector, q
     WHERE search_vector @@ q.query
     GROUP BY metadata->>'doc_id'
     ORDER BY max(ts_rank_cd(search_vector, q.query)) DESC, metadata->>'doc_id'
     LIMIT {top_k}
)
SELECT coalesce(json_agg(json_build_object('doc_id', doc_id, 'score', score)), '[]'::json)
  FROM hits;
"""
    completed = subprocess.run(
        ["docker", "exec", "-i", container, "psql", "-U", user, "-d", database, "-At", "-c", sql],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    output = completed.stdout.strip()
    return json.loads(output or "[]")


def _sample_metrics(expected: list[str], retrieved: list[str]) -> dict[str, float]:
    expected_set = set(expected)
    top5 = retrieved[:5]
    hit5 = float(bool(expected_set.intersection(top5)))
    recall5 = len(expected_set.intersection(top5)) / len(expected_set) if expected_set else 0.0
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved[:10], start=1):
        if doc_id in expected_set:
            reciprocal_rank = 1.0 / rank
            break
    return {"hit5": hit5, "recall5": recall5, "mrr10": reciprocal_rank}


def run(args: argparse.Namespace) -> dict[str, Any]:
    samples = load_samples(Path(args.dataset))
    if len(samples) > 20:
        raise RuntimeError(f"Refusing to run {len(samples)} samples; this script is limited to <=20.")
    ragent_to_biz = _load_doc_map(Path(args.doc_map))
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        started = time.perf_counter()
        if args.mode == "bm25":
            hits = _query_bm25(args.container, args.database, args.user, sample.query, args.top_k)
        else:
            hits = _query_fts(args.container, args.database, args.user, sample.query, args.top_k)
        wall_ms = (time.perf_counter() - started) * 1000
        retrieved_raw = [str(hit["doc_id"]) for hit in hits]
        retrieved = [ragent_to_biz.get(doc_id, doc_id) for doc_id in retrieved_raw]
        metrics = _sample_metrics(sample.expected_doc_ids, retrieved)
        row = {
            "query_id": sample.query_id,
            "query": sample.query,
            "expected_doc_ids": sample.expected_doc_ids,
            "retrieved_doc_ids": retrieved,
            "retrieved_doc_ids_raw": retrieved_raw,
            "scores": hits,
            "wall_ms": wall_ms,
            **metrics,
        }
        rows.append(row)
        print(
            f"[{index:>2}/{len(samples)}] {sample.query_id:<7} "
            f"hit5={int(metrics['hit5'])} recall5={metrics['recall5']:.3f} wall={wall_ms:.0f}ms"
        )
    summary = {
        "label": args.label,
        "mode": args.mode,
        "sample_count": len(rows),
        "top_k": args.top_k,
        "hit5": statistics.fmean(row["hit5"] for row in rows),
        "recall5": statistics.fmean(row["recall5"] for row in rows),
        "mrr10": statistics.fmean(row["mrr10"] for row in rows),
        "wall_p50_ms": _percentile([row["wall_ms"] for row in rows], 0.50),
        "wall_p95_ms": _percentile([row["wall_ms"] for row in rows], 0.95),
    }
    result = {"summary": summary, "samples": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output={output.resolve()}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--doc-map", default=str(DEFAULT_DOC_MAP))
    parser.add_argument("--container", default="postgres")
    parser.add_argument("--database", default="ragent")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", choices=("bm25", "fts"), default="bm25")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
