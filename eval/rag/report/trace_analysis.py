"""Aggregate retrieval trace stages for evaluation reports."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from eval.common.schemas import EvalRecord

STAGES = (
    "retrieval-scope-resolve",
    "vector-intent-search",
    "vector-global-search",
    "keyword-pg-search",
    "rrf-fusion",
    "chunk-deduplication",
    "rerank",
    "final-topk",
)
RETRIEVAL_THRESHOLD_MS = 2000


def analyze(records: list[EvalRecord]) -> dict[str, Any]:
    """Return stage latency, candidate, fallback, and ranking summaries."""
    stage_samples: dict[str, list[float]] = defaultdict(list)
    candidate_samples: dict[str, list[float]] = defaultdict(list)
    retrieval_samples: list[float] = []
    slowest: dict[str, Any] | None = None
    bottlenecks: list[dict[str, Any]] = []
    ranking_changes: list[dict[str, Any]] = []
    rerank_count = rerank_fallbacks = rerank_timeouts = 0

    for record in records:
        nodes = _nodes(record)
        per_stage: dict[str, float] = defaultdict(float)
        for node in nodes:
            name = node.get("nodeName")
            duration = float(node.get("durationMs") or 0)
            extra = _extra(node)
            if slowest is None or duration > slowest["duration_ms"]:
                slowest = {
                    "query_id": record.query_id,
                    "node": name or "?",
                    "duration_ms": duration,
                }
            if name == "multi-channel-retrieval":
                retrieval_samples.append(duration)
            if name not in STAGES:
                continue
            per_stage[name] += duration
            stage_samples[name].append(duration)
            candidate = extra.get("candidateCount", extra.get("inputCandidates"))
            if isinstance(candidate, (int, float)):
                candidate_samples[name].append(float(candidate))
            if name == "rerank":
                rerank_count += 1
                rerank_fallbacks += int(bool(extra.get("fallbackToRrf")))
                rerank_timeouts += int(bool(extra.get("timedOut")))
                for change in extra.get("rankingChanges") or []:
                    if isinstance(change, dict):
                        ranking_changes.append(
                            {"query_id": record.query_id, **change}
                        )
        retrieval_ms = sum(
            float(node.get("durationMs") or 0)
            for node in nodes
            if node.get("nodeName") == "multi-channel-retrieval"
        )
        if retrieval_ms > RETRIEVAL_THRESHOLD_MS:
            ranked = sorted(per_stage.items(), key=lambda item: item[1], reverse=True)
            bottlenecks.append(
                {
                    "query_id": record.query_id,
                    "retrieval_ms": retrieval_ms,
                    "bottleneck_stage": ranked[0][0] if ranked else "unknown",
                    "stage_ms": ranked[0][1] if ranked else 0,
                }
            )

    stage_rows = []
    for stage in STAGES:
        values = sorted(stage_samples.get(stage, []))
        candidates = candidate_samples.get(stage, [])
        if not values:
            continue
        stage_rows.append(
            {
                "stage": stage,
                "count": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "candidate_mean": (
                    sum(candidates) / len(candidates) if candidates else None
                ),
                "candidate_max": max(candidates) if candidates else None,
            }
        )

    return {
        "stages": stage_rows,
        "retrieval_latency": (
            {
                "count": len(retrieval_samples),
                "p50_ms": _percentile(sorted(retrieval_samples), 0.50),
                "p95_ms": _percentile(sorted(retrieval_samples), 0.95),
            }
            if retrieval_samples
            else None
        ),
        "slowest": slowest,
        "bottlenecks": bottlenecks,
        "rerank_count": rerank_count,
        "rerank_fallback_rate": (
            rerank_fallbacks / rerank_count if rerank_count else None
        ),
        "rerank_timeout_rate": (
            rerank_timeouts / rerank_count if rerank_count else None
        ),
        "ranking_changes": ranking_changes,
    }


def _nodes(record: EvalRecord) -> list[dict[str, Any]]:
    chat_nodes = _detail_nodes(record.chat_trace)
    eval_nodes = _detail_nodes(record.eval_trace)
    if _has_retrieval_stages(chat_nodes):
        return chat_nodes
    if _has_retrieval_stages(eval_nodes):
        non_retrieval_chat_nodes = [
            node
            for node in chat_nodes
            if node.get("nodeName") not in STAGES
            and node.get("nodeName") != "multi-channel-retrieval"
        ]
        return non_retrieval_chat_nodes + eval_nodes
    return chat_nodes or eval_nodes


def _detail_nodes(detail: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not detail or not detail.get("nodes"):
        return []
    return list(detail["nodes"])


def _has_retrieval_stages(nodes: list[dict[str, Any]]) -> bool:
    return any(node.get("nodeName") in STAGES for node in nodes)


def _extra(node: dict[str, Any]) -> dict[str, Any]:
    value = node.get("extraData")
    return value if isinstance(value, dict) else {}


def _percentile(values: list[float], quantile: float) -> float:
    return values[min(len(values) - 1, int(len(values) * quantile))]
