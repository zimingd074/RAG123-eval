"""Markdown / CSV / JSONL 报告产物。

从 ``score.py`` 落盘的 ``_scores.json`` + 原 ``runs/*.jsonl`` 重建报告：

    reports/<run>/report.md         一份完整报告（自建 + RAGAS 整体 + 按意图分层）
    reports/<run>/per_sample.csv    每行一条样本，所有指标横向铺开，方便分析
    reports/<run>/failures.jsonl    扩口径失败样例：
                                       - Hit@5 miss
                                       - answer_correctness < 0.5
                                       - 误拒（requires_rag=true 但 0 召回）
                                       - 过召回（requires_rag=false 却走 RAG）
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from eval.common.schemas import EvalRecord, MetricResult
from eval.rag.report.trace_analysis import analyze as analyze_traces

ANSWER_CORRECTNESS_FAIL_THRESHOLD = 0.5
FAITHFULNESS_FAIL_THRESHOLD = 0.5

# report.md 表里"自建指标"块的显示顺序
BASELINE_ORDER = [
    ("intent_top1", "意图 Top-1 准确率"),
    ("hit@1", "Hit@1"),
    ("hit@3", "Hit@3"),
    ("hit@5", "Hit@5"),
    ("hit@10", "Hit@10"),
    ("recall@5", "Recall@5"),
    ("recall_all_expected@5", "Recall@5 (must + nice)"),
    ("nice_hit@5", "Nice Hit@5"),
    ("recall@10", "Recall@10"),
    ("mrr@10", "MRR@10"),
    ("refusal_when_required", "误拒率（requires_rag 却 0 召回）"),
    ("fallback_when_required", "答案兜底率"),
    ("over_retrieval_rate", "过召回率（!requires_rag 却走 RAG）"),
    ("system_boundary_compliance", "SYSTEM Boundary Compliance"),
    ("ttft_p50_ms", "首字延迟 P50 (ms)"),
    ("ttft_p95_ms", "首字延迟 P95 (ms)"),
    ("ttft_mean_ms", "首字延迟均值 (ms)"),
    ("total_mean_ms", "整流均值 (ms)"),
]
RAGAS_ORDER = [
    ("faithfulness", "Faithfulness"),
    ("answer_relevancy", "Answer Relevancy"),
    ("answer_correctness", "Answer Correctness"),
    ("context_precision", "Context Precision"),
    ("context_recall", "Context Recall"),
]
RAGAS_KEYS = tuple(key for key, _ in RAGAS_ORDER)
MANUAL_SUFFIX = "_manual"

INTENT_BREAKDOWN_KEYS = [
    ("intent_top1", "Intent Top-1"),
    ("hit@5", "Hit@5"),
    ("recall@5", "Recall@5"),
    ("mrr@10", "MRR@10"),
    ("faithfulness", "Faithfulness"),
    ("answer_correctness", "Answer Correctness"),
]


def _fmt(m: MetricResult, v: float | None) -> str:
    if v is None:
        return "—"
    if m.name.endswith("_ms"):
        return f"{int(v)}"
    if m.is_pct:
        return f"{v * 100:.1f}%"
    return f"{v:.3f}"


def _fmt_csv_value(v: float | None) -> str:
    return "" if v is None else f"{v:.4f}"


def _manual_col(metric_name: str) -> str:
    return f"{metric_name}{MANUAL_SUFFIX}"


def _parse_manual_score(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("%"):
        value = float(text[:-1]) / 100
    else:
        value = float(text)
        if value > 1:
            value = value / 100
    if not 0 <= value <= 1:
        raise ValueError(f"人工分必须在 0-1 或 0-100 范围内：{raw!r}")
    return value


def load_manual_overrides(per_sample_path: Path) -> dict[str, dict[str, float]]:
    """从已有 per_sample.csv 读取人工列；没有文件或没有人工列时返回空。"""
    if not per_sample_path.exists():
        return {}

    overrides: dict[str, dict[str, float]] = {key: {} for key in RAGAS_KEYS}
    with per_sample_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        manual_cols = {key: _manual_col(key) for key in RAGAS_KEYS if _manual_col(key) in fields}
        if not manual_cols:
            return {}

        for row_no, row in enumerate(reader, start=2):
            qid = (row.get("query_id") or "").strip()
            if not qid:
                continue
            for key, col in manual_cols.items():
                try:
                    score = _parse_manual_score(row.get(col))
                except ValueError as exc:
                    raise ValueError(f"{per_sample_path}:{row_no}:{col} {exc}") from exc
                if score is not None:
                    overrides[key][qid] = score

    return {key: scores for key, scores in overrides.items() if scores}


def apply_manual_overrides(
    records: list[EvalRecord],
    metrics: list[MetricResult],
    overrides: dict[str, dict[str, float]],
) -> list[MetricResult]:
    """对 RAGAS 指标应用人工优先口径：人工列有值用人工，否则用原 RAGAS。"""
    if not overrides:
        return metrics

    updated: list[MetricResult] = []
    known_qids = {record.query_id for record in records}
    for metric in metrics:
        if metric.name not in RAGAS_KEYS:
            updated.append(metric)
            continue

        manual_scores = {
            qid: score
            for qid, score in overrides.get(metric.name, {}).items()
            if qid in known_qids
        }
        if not manual_scores:
            updated.append(metric)
            continue

        per_sample = dict(metric.per_sample)
        for qid, score in manual_scores.items():
            per_sample[qid] = score

        values: list[float] = []
        by_l1: dict[str, list[float]] = defaultdict(list)
        by_l2: dict[str, list[float]] = defaultdict(list)
        for record in records:
            value = per_sample.get(record.query_id)
            if value is None:
                continue
            values.append(value)
            by_l1[record.intent_l1 or "?"].append(value)
            by_l2[record.intent_l2 or "?"].append(value)

        def mean(xs: list[float]) -> float | None:
            return sum(xs) / len(xs) if xs else None

        meta = dict(metric.meta)
        meta["manual_overrides"] = len(manual_scores)
        meta["manual_policy"] = "manual value first, fallback to ragas"
        updated.append(
            MetricResult(
                name=metric.name,
                overall=mean(values),
                by_intent_l1={key: mean(vals) for key, vals in by_l1.items()},
                by_intent_l2={key: mean(vals) for key, vals in by_l2.items()},
                per_sample=per_sample,
                meta=meta,
                is_pct=metric.is_pct,
            )
        )
    return updated


def render_report_md(
    runs_file: Path,
    metrics: list[MetricResult],
    n_records: int,
    status: dict[str, int],
    run_metadata: dict | None = None,
    records: list[EvalRecord] | None = None,
) -> str:
    idx = {m.name: m for m in metrics}
    has_ragas = any(k in idx for k, _ in RAGAS_ORDER)

    lines: list[str] = []
    lines.append("# 评测报告\n")
    lines.append(f"> 数据源：`{runs_file.name}`")
    lines.append(f"> 样本数：{n_records}")
    lines.append(f"> 状态分布：{status}\n")
    run_metadata = run_metadata or {}
    if run_metadata:
        lines.append(f"> Profile：`{run_metadata.get('profile', '?')}`")
        lines.append(
            f"> 纳入/排除：{run_metadata.get('selected_sample_count', n_records)}"
            f" / {run_metadata.get('excluded_sample_count', 0)}"
        )
        lines.append(
            f"> 数据集 SHA-256：`{run_metadata.get('dataset_sha256', '?')}`\n"
        )

    lines.append("## 自建指标\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---|")
    for key, label in BASELINE_ORDER:
        if key in idx:
            lines.append(f"| {label} | {_fmt(idx[key], idx[key].overall)} |")

    if has_ragas:
        ragas_meta = next((m.meta for m in metrics if m.name in {k for k, _ in RAGAS_ORDER}), {})
        lines.append(
            f"\n## RAGAS LLM-as-judge "
            f"（评测 {ragas_meta.get('n_evaluable', '?')} 条，跳过 {ragas_meta.get('n_skipped', '?')} 条）\n"
        )
        if ragas_meta.get("judge_model"):
            lines.append(
                f"> judge：`{ragas_meta.get('judge_model')}`；"
                f"embedding：`{ragas_meta.get('embedding_model', '?')}`；"
                f"n_runs：{ragas_meta.get('n_runs', 1)}\n"
            )
        manual_count = sum(
            (idx[key].meta or {}).get("manual_overrides", 0)
            for key, _ in RAGAS_ORDER
            if key in idx
        )
        if manual_count:
            lines.append(
                f"> 口径：人工列优先，空值回退 RAGAS；"
                f"本次使用人工分 {manual_count} 个。\n"
            )
        lines.append("| 指标 | 数值 |")
        lines.append("|---|---|")
        for key, label in RAGAS_ORDER:
            if key in idx:
                lines.append(f"| {label} | {_fmt(idx[key], idx[key].overall)} |")

    lines.append("\n## 按 intent_l2 分层（核心指标）\n")
    all_intents: set[str] = set()
    for key, _ in INTENT_BREAKDOWN_KEYS:
        if key in idx:
            all_intents.update(idx[key].by_intent_l2.keys())

    if all_intents:
        header_cols = ["intent_l2"] + [label for key, label in INTENT_BREAKDOWN_KEYS if key in idx]
        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join("---" for _ in header_cols) + "|")
        for intent in sorted(all_intents):
            row = [intent]
            for key, _ in INTENT_BREAKDOWN_KEYS:
                if key not in idx:
                    continue
                row.append(_fmt(idx[key], idx[key].by_intent_l2.get(intent)))
            lines.append("| " + " | ".join(row) + " |")
    else:
        lines.append("_无样本_")

    if records:
        trace_summary = analyze_traces(records)
        if trace_summary["stages"]:
            lines.append("\n## Retrieval Trace\n")
            lines.append("| Stage | Count | P50 (ms) | P95 (ms) | Candidates avg/max |")
            lines.append("|---|---:|---:|---:|---:|")
            for stage in trace_summary["stages"]:
                candidate_text = "—"
                if stage["candidate_mean"] is not None:
                    candidate_text = (
                        f"{stage['candidate_mean']:.1f}/"
                        f"{int(stage['candidate_max'])}"
                    )
                lines.append(
                    f"| {stage['stage']} | {stage['count']} | "
                    f"{int(stage['p50_ms'])} | {int(stage['p95_ms'])} | "
                    f"{candidate_text} |"
                )
            retrieval_latency = trace_summary["retrieval_latency"]
            if retrieval_latency:
                lines.append(
                    f"\n> multi-channel-retrieval: "
                    f"count {retrieval_latency['count']}, "
                    f"P50 {int(retrieval_latency['p50_ms'])}ms, "
                    f"P95 {int(retrieval_latency['p95_ms'])}ms"
                )
            slowest = trace_summary["slowest"]
            if slowest:
                lines.append(
                    f"\n> Slowest node: `{slowest['query_id']}` / "
                    f"`{slowest['node']}` / {int(slowest['duration_ms'])}ms"
                )
            fallback = trace_summary["rerank_fallback_rate"]
            timeout = trace_summary["rerank_timeout_rate"]
            if fallback is not None:
                lines.append(
                    f"> Rerank fallback rate: {fallback * 100:.1f}%; "
                    f"timeout rate: {timeout * 100:.1f}%"
                )
            if trace_summary["bottlenecks"]:
                lines.append("\n### Retrieval Bottlenecks (>2000ms)\n")
                lines.append("| Query | Retrieval (ms) | Bottleneck stage | Stage (ms) |")
                lines.append("|---|---:|---|---:|")
                for item in trace_summary["bottlenecks"]:
                    lines.append(
                        f"| `{item['query_id']}` | {int(item['retrieval_ms'])} | "
                        f"{item['bottleneck_stage']} | {int(item['stage_ms'])} |"
                    )
            changes = trace_summary["ranking_changes"]
            if changes:
                lines.append("\n### RRF / Rerank Rank Changes\n")
                lines.append("| Query | Chunk | RRF rank | Rerank rank |")
                lines.append("|---|---|---:|---:|")
                for change in changes[:30]:
                    lines.append(
                        f"| `{change['query_id']}` | "
                        f"`{change.get('chunkId', '?')}` | "
                        f"{change.get('rrfRank', '—')} | "
                        f"{change.get('rerankRank', '—')} |"
                    )

        lines.append("\n## 按难度分层（核心指标）\n")
        difficulty_rows = _difficulty_breakdown(records, idx)
        if difficulty_rows:
            labels = [
                label
                for key, label in INTENT_BREAKDOWN_KEYS
                if key in idx
            ]
            lines.append("| difficulty | " + " | ".join(labels) + " |")
            lines.append("|---|" + "|".join("---" for _ in labels) + "|")
            for difficulty in ("easy", "medium", "hard"):
                if difficulty not in difficulty_rows:
                    continue
                values = difficulty_rows[difficulty]
                lines.append(
                    "| "
                    + difficulty
                    + " | "
                    + " | ".join(values)
                    + " |"
                )

    deferred_ids = run_metadata.get("excluded_sample_ids", [])
    if deferred_ids:
        lines.append("\n## Tool-deferred 样本\n")
        lines.append(f"> 共 {len(deferred_ids)} 条，不进入 static-v1 核心指标。\n")
        excluded_samples = run_metadata.get("excluded_samples") or []
        if excluded_samples:
            by_intent: dict[str, list[str]] = defaultdict(list)
            for sample in excluded_samples:
                by_intent[sample.get("intent_l2", "?")].append(
                    sample.get("query_id", "?")
                )
            lines.append("| intent_l2 | 样本 ID |")
            lines.append("|---|---|")
            for intent_l2, query_ids in sorted(by_intent.items()):
                lines.append(
                    f"| {intent_l2} | "
                    + ", ".join(f"`{query_id}`" for query_id in query_ids)
                    + " |"
                )
        else:
            lines.append(", ".join(f"`{query_id}`" for query_id in deferred_ids))

    return "\n".join(lines) + "\n"


def _difficulty_breakdown(
    records: list[EvalRecord],
    metrics: dict[str, MetricResult],
) -> dict[str, list[str]]:
    """Aggregate displayed core metrics by sample difficulty."""
    output: dict[str, list[str]] = {}
    for difficulty in {record.difficulty for record in records}:
        values: list[str] = []
        for key, _ in INTENT_BREAKDOWN_KEYS:
            metric = metrics.get(key)
            if metric is None:
                continue
            bucket = [
                metric.per_sample.get(record.query_id)
                for record in records
                if record.difficulty == difficulty
                and metric.per_sample.get(record.query_id) is not None
            ]
            mean = sum(bucket) / len(bucket) if bucket else None
            values.append(_fmt(metric, mean))
        output[difficulty] = values
    return output


def render_per_sample_csv(
    records: list[EvalRecord],
    metrics: list[MetricResult],
    manual_overrides: dict[str, dict[str, float]] | None = None,
) -> list[list[str]]:
    """每条样本一行，所有指标的 per_sample 横向铺开。"""
    idx = {m.name: m for m in metrics}
    # 只保留有 per_sample 数据的指标（跳过仅聚合的，如 ttft_p50/p95/p99/mean/total_p95）
    metric_names = [m.name for m in metrics if m.per_sample]
    manual_overrides = manual_overrides or {}

    header = [
        "query_id",
        "intent_l1",
        "intent_l2",
        "difficulty",
        "requires_rag",
        "expected_route",
        "evaluation_scope",
        "requires_tool",
        "scope_reason",
        "final_status",
        "chat_trace_id",
        "eval_trace_id",
    ]
    for name in metric_names:
        header.append(name)
        if name in RAGAS_KEYS:
            header.append(_manual_col(name))

    rows: list[list[str]] = [header]
    for r in records:
        row = [
            r.query_id,
            r.intent_l1,
            r.intent_l2,
            r.difficulty,
            str(r.requires_rag),
            r.expected_route,
            r.evaluation_scope,
            str(r.requires_tool),
            r.scope_reason,
            r.final_status,
            r.chat_trace_id or "",
            r.eval_trace_id or "",
        ]
        for name in metric_names:
            v = idx[name].per_sample.get(r.query_id)
            row.append(_fmt_csv_value(v) if isinstance(v, float) or v is None else str(v))
            if name in RAGAS_KEYS:
                row.append(_fmt_csv_value(manual_overrides.get(name, {}).get(r.query_id)))
        rows.append(row)
    return rows


def get_failure_qids(
    records: list[EvalRecord],
    metrics: list[MetricResult],
) -> dict[str, list[str]]:
    """扩口径失败检测：Hit@5 miss / correctness 低 / 误拒 / 过召回。

    返回 {query_id: [reason, ...]}，与 render_failures 口径一致。
    """
    idx = {m.name: m for m in metrics}
    hit5 = idx.get("hit@5")
    correctness = idx.get("answer_correctness")
    faithfulness = idx.get("faithfulness")
    refusal = idx.get("refusal_when_required")
    over = idx.get("over_retrieval_rate")
    boundary = idx.get("system_boundary_compliance")
    intent = idx.get("intent_top1")
    ttft = idx.get("ttft_mean_ms")

    reasons_by_qid: dict[str, list[str]] = defaultdict(list)

    for r in records:
        if r.evaluation_scope == "tool-deferred":
            reasons_by_qid[r.query_id].append("tool_deferred")
            continue
        if intent is not None and intent.per_sample.get(r.query_id) == 0:
            reasons_by_qid[r.query_id].append("intent")
        if hit5 is not None:
            v = hit5.per_sample.get(r.query_id)
            if v == 0:
                reasons_by_qid[r.query_id].append("hit@5_miss")
        if correctness is not None:
            v = correctness.per_sample.get(r.query_id)
            if v is not None and v < ANSWER_CORRECTNESS_FAIL_THRESHOLD:
                reasons_by_qid[r.query_id].append(f"answer_correctness_low({v:.2f})")
        if faithfulness is not None:
            v = faithfulness.per_sample.get(r.query_id)
            if v is not None and v < FAITHFULNESS_FAIL_THRESHOLD:
                reasons_by_qid[r.query_id].append(f"faithfulness_low({v:.2f})")
        if refusal is not None:
            v = refusal.per_sample.get(r.query_id)
            if v == 1:
                reasons_by_qid[r.query_id].append("refused_when_required")
        if over is not None:
            v = over.per_sample.get(r.query_id)
            if v == 1:
                reasons_by_qid[r.query_id].append("over_retrieved")
        if boundary is not None and boundary.per_sample.get(r.query_id) == 0:
            reasons_by_qid[r.query_id].append("system_behavior")
        if ttft is not None:
            value = ttft.per_sample.get(r.query_id)
            if value is not None and value > 6000:
                reasons_by_qid[r.query_id].append("latency")

    return reasons_by_qid


def render_failures(
    records: list[EvalRecord],
    metrics: list[MetricResult],
) -> list[dict]:
    """扩口径失败：Hit@5 miss / correctness 低 / 误拒 / 过召回。每条样本最多一条记录，多个原因合并。"""
    reasons_by_qid = get_failure_qids(records, metrics)
    idx = {m.name: m for m in metrics}

    # 收集失败样本的分数快照
    score_by_qid: dict[str, dict] = defaultdict(dict)
    for r in records:
        if r.query_id not in reasons_by_qid:
            continue
        hit5 = idx.get("hit@5")
        if hit5 is not None and hit5.per_sample.get(r.query_id) == 0:
            score_by_qid[r.query_id]["hit@5"] = 0
        correctness = idx.get("answer_correctness")
        if correctness is not None:
            v = correctness.per_sample.get(r.query_id)
            if v is not None and v < ANSWER_CORRECTNESS_FAIL_THRESHOLD:
                score_by_qid[r.query_id]["answer_correctness"] = v
        faithfulness = idx.get("faithfulness")
        if faithfulness is not None:
            v = faithfulness.per_sample.get(r.query_id)
            if v is not None and v < FAITHFULNESS_FAIL_THRESHOLD:
                score_by_qid[r.query_id]["faithfulness"] = v

    failures: list[dict] = []
    for r in records:
        if r.query_id not in reasons_by_qid:
            continue
        failures.append(
            {
                "query_id": r.query_id,
                "intent_l1": r.intent_l1,
                "intent_l2": r.intent_l2,
                "requires_rag": r.requires_rag,
                "user_input": r.user_input,
                "reference_doc_ids": r.reference_doc_ids,
                "retrieved_doc_ids": r.retrieved_doc_ids,
                "first_token_ms": r.first_token_ms,
                "latency_ms": r.latency_ms,
                "final_status": r.final_status,
                "response_preview": (r.response or "")[:200],
                "failure_reasons": reasons_by_qid[r.query_id],
                "failure_categories": sorted(
                    {
                        _failure_category(reason)
                        for reason in reasons_by_qid[r.query_id]
                    }
                ),
                "scores": score_by_qid.get(r.query_id, {}),
            }
        )
    return failures


def _failure_category(reason: str) -> str:
    """Map concrete failure reasons to the plan's root-cause taxonomy."""
    if reason == "tool_deferred":
        return "tool_deferred"
    if reason == "intent":
        return "intent"
    if reason in {"over_retrieved", "system_behavior"}:
        return "system_behavior"
    if reason.startswith("hit@") or reason == "refused_when_required":
        return "retrieval"
    if reason == "latency":
        return "latency"
    return "generation"


def write_all(
    report_dir: Path,
    runs_file: Path,
    records: list[EvalRecord],
    metrics: list[MetricResult],
    status: dict[str, int],
    report_metrics: list[MetricResult] | None = None,
    manual_overrides: dict[str, dict[str, float]] | None = None,
    run_metadata: dict | None = None,
) -> dict[str, Path]:
    """写 report.md + per_sample.csv + failures.jsonl。返回 {产物名 → 路径}。"""
    report_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    report_metrics = report_metrics or metrics
    manual_overrides = manual_overrides or {}

    md_path = report_dir / "report.md"
    md_path.write_text(
        render_report_md(
            runs_file,
            report_metrics,
            len(records),
            status,
            run_metadata=run_metadata,
            records=records,
        ),
        encoding="utf-8",
    )
    out["report.md"] = md_path

    csv_path = report_dir / "per_sample.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(render_per_sample_csv(records, metrics, manual_overrides))
    out["per_sample.csv"] = csv_path

    failures_path = report_dir / "failures.jsonl"
    failures = render_failures(records, report_metrics)
    with failures_path.open("w", encoding="utf-8") as f:
        for rec in failures:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out["failures.jsonl"] = failures_path

    return out
