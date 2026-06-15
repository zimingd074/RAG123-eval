"""Two-stage routing-model and answer-model benchmark for ragent."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
from collections import Counter
from datetime import datetime
from functools import cmp_to_key
from html import escape
from pathlib import Path
from typing import Any

from eval.common.schemas import EvalRecord
from eval.rag.dataset.profiles import dataset_sha256, write_run_metadata
from eval.rag.metrics import rewrite as rewrite_metrics
from eval.rag.pipeline.runner import (
    DEFAULT_BASE_URL,
    _trace_node_extra,
    login,
    replay_generation,
    run as run_pipeline,
)
from eval.rag.pipeline.score import REPORTS_DIR, load_records, score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    PROJECT_ROOT / "eval" / "rag" / "dataset" / "eval_set_chat_models20_20260615.jsonl"
)
DEFAULT_CANDIDATES = (
    "qwen3-max",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-flash",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
)
BASELINE = "qwen3-max"


def percentile_95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return ordered[index]


def routing_metrics(
    records: list[EvalRecord],
    rewrite_reviews: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not records:
        return {
            "intent_accuracy": 0.0,
            "document_recall": 0.0,
            "rewrite_parse_rate": 0.0,
            "rewrite_semantic_score": 0.0,
            "silent_condition_changes": 0,
            "human_review_complete": False,
            "rewrite_metric_complete": False,
            "rewrite_metric_source": "automatic",
            "rewrite_change_details": {},
            "stability": 0.0,
            "quality": 0.0,
            "mean_input_tokens": 0.0,
            "mean_output_tokens": 0.0,
            "mean_latency_ms": 0.0,
        }
    intent_accuracy = sum(r.intent_pred == r.intent_l2 for r in records) / len(records)
    recalls = []
    for record in records:
        expected = set(record.reference_doc_ids)
        recalls.append(
            len(expected & set(record.retrieved_doc_ids)) / len(expected)
            if expected
            else 1.0
        )
    automatic_results = {
        metric.name: metric for metric in rewrite_metrics.compute(records)
    }
    rewrite_parse_rate = (
        automatic_results["rewrite_parse_rate"].overall or 0.0
    )
    automatic_semantic_score = (
        automatic_results["rewrite_semantic_preservation"].overall or 0.0
    )
    automatic_change_metric = automatic_results["silent_condition_change_rate"]
    automatic_evaluated_count = int(
        automatic_change_metric.meta.get("evaluated_count") or 0
    )
    rewrite_metric_complete = automatic_evaluated_count == len(records)
    review_rows = rewrite_reviews or {}
    human_review_complete = bool(records) and all(
        record.query_id in review_rows
        and isinstance(review_rows[record.query_id].get("semantic_score"), (int, float))
        and not isinstance(review_rows[record.query_id].get("semantic_score"), bool)
        and 0 <= float(review_rows[record.query_id]["semantic_score"]) <= 1
        and isinstance(
            review_rows[record.query_id].get("silent_condition_change"),
            bool,
        )
        for record in records
    )
    rewrite_semantic_score = (
        statistics.fmean(
            float(review_rows[record.query_id]["semantic_score"])
            for record in records
        )
        if human_review_complete
        else automatic_semantic_score
    )
    rewrite_metric_complete = human_review_complete or rewrite_metric_complete
    silent_condition_changes = (
        sum(
            bool(review_rows[record.query_id]["silent_condition_change"])
            for record in records
        )
        if human_review_complete
        else int(automatic_change_metric.meta.get("changed_count") or 0)
    )
    stability = sum(
        record.final_status == "success"
        and not any(bool(call.get("fallback")) for call in record.model_calls)
        for record in records
    ) / len(records)
    document_recall = statistics.fmean(recalls)
    quality = (
        intent_accuracy * 0.45
        + document_recall * 0.30
        + rewrite_semantic_score * 0.15
        + stability * 0.10
    )
    routing_steps = {"query-rewrite", "intent-classify", "ambiguity-check"}
    input_tokens = []
    output_tokens = []
    latencies = []
    for record in records:
        calls = [
            call
            for call in record.model_calls
            if call.get("step") in routing_steps
        ]
        input_tokens.append(
            sum(float(call.get("estimatedInputTokens") or 0) for call in calls)
        )
        output_tokens.append(
            sum(float(call.get("estimatedOutputTokens") or 0) for call in calls)
        )
        latencies.append(sum(float(call.get("latencyMs") or 0) for call in calls))
    return {
        "intent_accuracy": intent_accuracy,
        "document_recall": document_recall,
        "rewrite_parse_rate": rewrite_parse_rate,
        "rewrite_semantic_score": rewrite_semantic_score,
        "silent_condition_changes": silent_condition_changes,
        "human_review_complete": human_review_complete,
        "rewrite_metric_complete": rewrite_metric_complete,
        "rewrite_metric_source": (
            "human_review" if human_review_complete else "automatic"
        ),
        "rewrite_change_details": (
            {}
            if human_review_complete
            else automatic_change_metric.meta.get("changes") or {}
        ),
        "stability": stability,
        "quality": quality,
        "mean_input_tokens": statistics.fmean(input_tokens),
        "mean_output_tokens": statistics.fmean(output_tokens),
        "mean_latency_ms": statistics.fmean(latencies),
    }


def write_rewrite_review_template(
    path: Path,
    routing_runs: dict[str, Path],
) -> None:
    template: dict[str, dict[str, Any]] = {}
    for model, runs_file in routing_runs.items():
        rows: dict[str, Any] = {}
        for record in load_records(runs_file):
            snapshot = _trace_node_extra(record.chat_trace, "eval-query-rewrite")
            rows[record.query_id] = {
                "original_question": snapshot.get("originalQuestion") or record.user_input,
                "rewritten_question": snapshot.get("rewrittenQuestion"),
                "sub_questions": snapshot.get("subQuestions") or [],
                "semantic_score": None,
                "silent_condition_change": None,
                "review_note": "",
            }
        template[model] = rows
    path.write_text(
        json.dumps(template, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def generation_messages(record: EvalRecord) -> list[dict[str, Any]]:
    snapshot = _trace_node_extra(record.chat_trace, "eval-answer-generation")
    return list(snapshot.get("messages") or [])


def write_records(path: Path, records: list[EvalRecord], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    write_run_metadata(path, metadata)


def replay_answer_arm(
    *,
    source_records: list[EvalRecord],
    routing_model: str,
    answer_model: str,
    base_url: str,
    token: str,
    out_path: Path,
    dataset: Path,
) -> Path:
    replayed: list[EvalRecord] = []
    for index, record in enumerate(source_records, start=1):
        messages = generation_messages(record)
        if not messages:
            updated_calls = list(record.model_calls)
            updated_calls.append(
                {
                    "step": "answer-replay",
                    "modelId": answer_model,
                    "parsed": False,
                    "fallback": True,
                    "skipReason": "missing_frozen_generation_messages",
                    "estimatedInputTokens": 0,
                    "estimatedOutputTokens": 0,
                }
            )
            replayed.append(
                dataclasses.replace(
                    record,
                    answer_model_id=answer_model,
                    model_calls=updated_calls,
                )
            )
            print(
                f"    [{index:02d}/{len(source_records)}] {record.query_id} "
                "skipped: no frozen generation messages"
            )
            continue
        state = replay_generation(base_url, token, messages, answer_model)
        input_text = "\n".join(
            f"{item.get('role', '')}\n{item.get('content', '')}" for item in messages
        )
        updated_calls = list(record.model_calls)
        updated_calls.append(
            {
                "step": "answer-replay",
                "modelId": answer_model,
                "parsed": state["final_status"] == "success",
                "fallback": False,
                "estimatedInputTokens": (len(input_text) + 1) // 2,
                "estimatedOutputTokens": (len(state["response"]) + 1) // 2,
            }
        )
        replayed.append(
            dataclasses.replace(
                record,
                response=state["response"],
                thinking=None,
                latency_ms=state["latency_ms"],
                first_token_ms=state["first_token_ms"],
                final_status=state["final_status"],
                error=state["error"],
                answer_model_id=answer_model,
                estimated_input_tokens=(len(input_text) + 1) // 2,
                estimated_output_tokens=(len(state["response"]) + 1) // 2,
                usage_estimated=True,
                model_calls=updated_calls,
            )
        )
        print(
            f"    [{index:02d}/{len(source_records)}] {record.query_id} "
            f"{state['final_status']} {state['latency_ms']}ms"
        )
    write_records(
        out_path,
        replayed,
        {
            "benchmark": "chat-models",
            "mode": "frozen-answer-replay",
            "dataset_path": str(dataset.resolve()),
            "dataset_sha256": dataset_sha256(dataset),
            "routing_model": routing_model,
            "answer_model": answer_model,
            "source_run": str(out_path.parent / f"routing__{routing_model}.jsonl"),
            "selected_sample_count": len(replayed),
            "profile": "static-v1",
        },
    )
    return out_path


def load_score_summary(runs_file: Path) -> dict[str, Any]:
    scores_file = REPORTS_DIR / runs_file.stem / "_scores.json"
    payload = json.loads(scores_file.read_text(encoding="utf-8"))
    metrics = {item["name"]: item for item in payload["metrics"]}
    records = load_records(runs_file)
    hard_ids = {record.query_id for record in records if record.difficulty == "hard"}
    hard_values = [
        value
        for query_id, value in (metrics.get("answer_correctness", {}).get("per_sample") or {}).items()
        if query_id in hard_ids and value is not None
    ]
    ttft_values = [
        r.first_token_ms for r in records if r.first_token_ms is not None
    ]
    latency_values = [r.latency_ms for r in records]
    breakdowns: dict[str, dict[str, dict[str, float | None]]] = {}
    dimensions = {
        "difficulty": lambda record: record.difficulty,
        "intent_l2": lambda record: record.intent_l2,
        "answer_type": lambda record: record.expected_answer_type or "unknown",
    }
    for dimension, group_key in dimensions.items():
        groups: dict[str, list[EvalRecord]] = {}
        for record in records:
            groups.setdefault(group_key(record), []).append(record)
        breakdowns[dimension] = {}
        for group, group_records in groups.items():
            query_ids = {record.query_id for record in group_records}
            group_metrics: dict[str, float | None] = {}
            for metric_name in (
                "answer_correctness",
                "faithfulness",
                "answer_relevancy",
            ):
                per_sample = metrics.get(metric_name, {}).get("per_sample") or {}
                values = [
                    float(value)
                    for query_id, value in per_sample.items()
                    if query_id in query_ids and value is not None
                ]
                group_metrics[metric_name] = (
                    statistics.fmean(values) if values else None
                )
            breakdowns[dimension][group] = group_metrics
    return {
        "answer_correctness": metrics.get("answer_correctness", {}).get("overall"),
        "faithfulness": metrics.get("faithfulness", {}).get("overall"),
        "answer_relevancy": metrics.get("answer_relevancy", {}).get("overall"),
        "hard_answer_correctness": (
            statistics.fmean(hard_values) if hard_values else None
        ),
        "success_rate": sum(r.final_status == "success" for r in records) / len(records),
        "mean_ttft_ms": statistics.fmean(ttft_values) if ttft_values else None,
        "p95_ttft_ms": percentile_95(ttft_values),
        "mean_latency_ms": statistics.fmean(r.latency_ms for r in records),
        "p95_latency_ms": percentile_95(latency_values),
        "mean_input_tokens": statistics.fmean(
            r.estimated_input_tokens or 0 for r in records
        ),
        "mean_output_tokens": statistics.fmean(
            r.estimated_output_tokens or 0 for r in records
        ),
        "breakdowns": breakdowns,
    }


def score_is_reusable(
    runs_file: Path,
    *,
    judge_model: str,
    judge_base_url: str,
    skip_ragas: bool,
) -> bool:
    scores_file = REPORTS_DIR / runs_file.stem / "_scores.json"
    if not scores_file.exists():
        return False
    if skip_ragas:
        return True
    payload = json.loads(scores_file.read_text(encoding="utf-8"))
    metrics = {item["name"]: item for item in payload.get("metrics") or []}
    for name in ("answer_correctness", "faithfulness", "answer_relevancy"):
        metric = metrics.get(name)
        if not metric or metric.get("overall") is None:
            return False
        meta = metric.get("meta") or {}
        if meta.get("judge_model") != judge_model:
            return False
        if meta.get("judge_base_url", "").rstrip("/") != judge_base_url.rstrip("/"):
            return False
    return True


def answer_quality(summary: dict[str, Any]) -> float | None:
    values = (
        summary.get("answer_correctness"),
        summary.get("faithfulness"),
        summary.get("answer_relevancy"),
    )
    if any(value is None for value in values):
        return None
    return values[0] * 0.45 + values[1] * 0.35 + values[2] * 0.20


def model_cost_per_1000(
    input_tokens: float,
    output_tokens: float,
    price: dict[str, Any] | None,
) -> float | None:
    if not price:
        return None
    input_rate = price.get("input_per_million")
    output_rate = price.get("output_per_million")
    if input_rate is None or output_rate is None:
        return None
    return (
        input_tokens * float(input_rate)
        + output_tokens * float(output_rate)
    ) / 1000


def cost_per_1000(
    routing: dict[str, Any],
    answer: dict[str, Any],
    routing_price: dict[str, Any] | None,
    answer_price: dict[str, Any] | None,
) -> float | None:
    routing_cost = model_cost_per_1000(
        routing["mean_input_tokens"],
        routing["mean_output_tokens"],
        routing_price,
    )
    answer_cost = model_cost_per_1000(
        answer["mean_input_tokens"],
        answer["mean_output_tokens"],
        answer_price,
    )
    if routing_cost is None or answer_cost is None:
        return None
    return routing_cost + answer_cost


def average_metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = set().union(*(item.keys() for item in items))
    result: dict[str, Any] = {}
    for key in keys:
        present = [item.get(key) for item in items if key in item]
        if present and all(isinstance(value, dict) for value in present):
            result[key] = average_metrics(present)
            continue
        if present and all(isinstance(value, bool) for value in present):
            result[key] = all(present)
            continue
        values = [
            float(item[key])
            for item in items
            if isinstance(item.get(key), (int, float))
            and not isinstance(item.get(key), bool)
        ]
        result[key] = statistics.fmean(values) if values else items[0].get(key)
    return result


def write_pareto_svg(path: Path, arms: list[dict[str, Any]]) -> None:
    points = [
        arm
        for arm in arms
        if arm.get("cost_per_1000") is not None
        and arm.get("answer_quality") is not None
    ]
    width, height = 1000, 620
    margin = 80
    if not points:
        path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="200">'
            '<text x="30" y="80">Pareto chart unavailable: missing prices or scores.</text>'
            "</svg>",
            encoding="utf-8",
        )
        return
    costs = [float(arm["cost_per_1000"]) for arm in points]
    qualities = [float(arm["answer_quality"]) for arm in points]
    min_cost, max_cost = min(costs), max(costs)
    min_quality, max_quality = min(qualities), max(qualities)

    def x_pos(value: float) -> float:
        span = max_cost - min_cost
        return margin + (value - min_cost) / (span or 1) * (width - margin * 2)

    def y_pos(value: float) -> float:
        span = max_quality - min_quality
        return height - margin - (value - min_quality) / (span or 1) * (
            height - margin * 2
        )

    frontier = [
        arm
        for arm in points
        if not any(
            other is not arm
            and other["cost_per_1000"] <= arm["cost_per_1000"]
            and other["answer_quality"] >= arm["answer_quality"]
            and (
                other["cost_per_1000"] < arm["cost_per_1000"]
                or other["answer_quality"] > arm["answer_quality"]
            )
            for other in points
        )
    ]
    frontier.sort(key=lambda arm: arm["cost_per_1000"])
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" '
        f'y2="{height - margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" '
        f'y2="{height - margin}" stroke="#333"/>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle">'
        "CNY / 1000 complete questions</text>",
        f'<text x="24" y="{height / 2}" transform="rotate(-90 24 {height / 2})" '
        'text-anchor="middle">Answer quality</text>',
    ]
    if len(frontier) > 1:
        coordinates = " ".join(
            f"{x_pos(float(arm['cost_per_1000']))},{y_pos(float(arm['answer_quality']))}"
            for arm in frontier
        )
        svg.append(
            f'<polyline points="{coordinates}" fill="none" '
            'stroke="#d97706" stroke-width="3"/>'
        )
    for arm in points:
        x = x_pos(float(arm["cost_per_1000"]))
        y = y_pos(float(arm["answer_quality"]))
        color = "#15803d" if arm.get("eligible") else "#64748b"
        label = escape(f"{arm['routing_model']} + {arm['answer_model']}")
        svg.extend([
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>',
            f'<text x="{x + 9:.1f}" y="{y - 7:.1f}" font-size="12">{label}</text>',
        ])
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def normalized_lower(values: list[float | None], value: float | None) -> float:
    available = [item for item in values if item is not None]
    if value is None or not available:
        return 0.0
    low, high = min(available), max(available)
    return 1.0 if high == low else (high - value) / (high - low)


def final_rank(
    arms: list[dict[str, Any]],
    baseline: dict[str, Any],
) -> list[dict[str, Any]]:
    costs = [arm.get("cost_per_1000") for arm in arms]
    ttfts = [arm["answer"].get("p95_ttft_ms") for arm in arms]
    latencies = [arm["answer"].get("p95_latency_ms") for arm in arms]
    for arm in arms:
        answer = arm["answer"]
        quality = arm.get("answer_quality")
        gates = {
            "faithfulness": (
                answer.get("faithfulness") is not None
                and baseline["answer"].get("faithfulness") is not None
                and answer["faithfulness"] >= baseline["answer"]["faithfulness"] - 0.02
            ),
            "hard_quality": (
                answer.get("hard_answer_correctness") is not None
                and baseline["answer"].get("hard_answer_correctness") is not None
                and answer["hard_answer_correctness"]
                >= baseline["answer"]["hard_answer_correctness"] - 0.02
            ),
            "intent": (
                arm["routing"]["intent_accuracy"]
                >= baseline["routing"]["intent_accuracy"] - 0.05
            ),
            "success": answer["success_rate"] >= 0.99,
        }
        arm["gates"] = gates
        arm["eligible"] = all(gates.values())
        arm["final_score"] = (
            (quality or 0.0) * 0.60
            + arm["routing"]["quality"] * 0.20
            + normalized_lower(costs, arm.get("cost_per_1000")) * 0.15
            + (
                normalized_lower(ttfts, answer.get("p95_ttft_ms")) * 0.35
                + normalized_lower(
                    latencies,
                    answer.get("p95_latency_ms"),
                )
                * 0.35
                + answer["success_rate"] * 0.30
            )
            * 0.05
        )
    def compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        if left["eligible"] != right["eligible"]:
            return -1 if left["eligible"] else 1
        score_delta = left["final_score"] - right["final_score"]
        if abs(score_delta) > 0.01:
            return -1 if score_delta > 0 else 1
        left_cost = left.get("cost_per_1000")
        right_cost = right.get("cost_per_1000")
        left_cost = float("inf") if left_cost is None else left_cost
        right_cost = float("inf") if right_cost is None else right_cost
        if left_cost != right_cost:
            return -1 if left_cost < right_cost else 1
        left_ttft = left["answer"].get("p95_ttft_ms")
        right_ttft = right["answer"].get("p95_ttft_ms")
        left_ttft = float("inf") if left_ttft is None else left_ttft
        right_ttft = float("inf") if right_ttft is None else right_ttft
        if left_ttft == right_ttft:
            return 0
        return -1 if left_ttft < right_ttft else 1

    return sorted(arms, key=cmp_to_key(compare))


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# RAG 双模型选型结果",
        "",
        f"- 数据集：`{result['dataset']}`",
        f"- 候选：{', '.join(result['candidates'])}",
        f"- Judge：`gpt-5.4-mini`",
    ]
    recommendation = result.get("recommendation")
    backup = result.get("low_cost_backup")
    if recommendation:
        lines.append(
            f"- 推荐：`{recommendation['routing_model']} + "
            f"{recommendation['answer_model']}`"
        )
    if backup:
        lines.append(
            f"- 合格备选：`{backup['routing_model']} + "
            f"{backup['answer_model']}`"
        )
    rewrite_review_complete = all(
        item["metrics"].get("human_review_complete")
        for item in result["routing_ranking"]
    )
    if not rewrite_review_complete:
        lines.append(
            "- 改写复核：未完成人工复核，当前使用离线条件保真与字符二元组指标"
        )
    lines.extend([
        "",
        "## 前置模型",
        "",
        "| 排名 | 模型 | 综合 | 意图准确率 | 文档召回 | 改写语义 | 静默改写 | 稳定率 | 成本/千次 | 前置耗时 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for index, item in enumerate(result["routing_ranking"], start=1):
        metrics = item["metrics"]
        lines.append(
            f"| {index} | {item['model']} | {metrics['quality']:.4f} | "
            f"{metrics['intent_accuracy']:.2%} | {metrics['document_recall']:.2%} | "
            f"{metrics['rewrite_semantic_score']:.2%} | "
            f"{metrics['silent_condition_changes']} | {metrics['stability']:.2%} | "
            f"{item.get('cost_per_1000') if item.get('cost_per_1000') is not None else 'N/A'} | "
            f"{metrics['mean_latency_ms']:.0f} ms |"
        )
    lines.extend(
        [
            "",
            "## 模型组合",
            "",
            "| 排名 | 前置模型 | 回答模型 | 合格 | 综合 | 回答质量 | 正确性 | 忠实性 | 相关性 | 成本/千次 | P95 TTFT |",
            "|---:|---|---|:---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for index, arm in enumerate(result["combination_ranking"], start=1):
        ttft = arm["answer"].get("p95_ttft_ms")
        ttft_text = f"{ttft:.0f} ms" if ttft is not None else "N/A"
        lines.append(
            f"| {index} | {arm['routing_model']} | {arm['answer_model']} | "
            f"{'是' if arm['eligible'] else '否'} | {arm['final_score']:.4f} | "
            f"{(arm.get('answer_quality') or 0):.4f} | "
            f"{arm['answer'].get('answer_correctness') if arm['answer'].get('answer_correctness') is not None else 'N/A'} | "
            f"{arm['answer'].get('faithfulness') if arm['answer'].get('faithfulness') is not None else 'N/A'} | "
            f"{arm['answer'].get('answer_relevancy') if arm['answer'].get('answer_relevancy') is not None else 'N/A'} | "
            f"{arm.get('cost_per_1000') if arm.get('cost_per_1000') is not None else 'N/A'} | "
            f"{ttft_text} |"
        )
    lines.extend(
        [
            "",
            "> Token 在供应商未返回 usage 时为估算值；最终上线前仍需完成人工改写语义复核。",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    candidates = tuple(
        item.strip() for item in args.candidates.split(",") if item.strip()
    )
    if BASELINE not in candidates:
        raise RuntimeError(f"candidate list must contain baseline {BASELINE}")
    if args.final_rounds < 1 or args.finalists < 1 or args.top_routing < 1:
        raise RuntimeError("top-routing, finalists and final-rounds must be positive")
    judge_model = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")
    judge_base_url = os.environ.get(
        "JUDGE_BASE_URL",
        "https://api.86gamestore.com/responses",
    )
    if not args.skip_ragas and judge_model != "gpt-5.4-mini":
        raise RuntimeError(
            f"JUDGE_MODEL must be gpt-5.4-mini for this benchmark, got {judge_model}"
        )
    manifest_path = args.dataset.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("human_review_status") != "approved"
            and not args.allow_pending_dataset_review
        ):
            raise RuntimeError(
                f"dataset review is not approved: {manifest_path}; "
                "review the 20 samples or pass --allow-pending-dataset-review"
            )
    output = (
        args.output
        or (
            PROJECT_ROOT
            / "eval"
            / "reports"
            / f"chat_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    ).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs_dir = output / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    catalog = (
        json.loads(args.prices.read_text(encoding="utf-8"))
        if args.prices
        else {}
    )
    prices = catalog.get("models", catalog)
    required_catalog_fields = (
        "actual_version",
        "context_limit",
        "input_per_million",
        "output_per_million",
    )
    missing_prices = [
        model
        for model in candidates
        if model not in prices
        or any(prices[model].get(field) is None for field in required_catalog_fields)
    ]
    if missing_prices and not args.allow_missing_prices:
        raise RuntimeError(
            "incomplete execution-day model catalog for: "
            + ", ".join(missing_prices)
            + "; pass --prices <catalog.json>"
        )
    today = datetime.now().date().isoformat()
    if (
        catalog
        and catalog.get("snapshot_date") != today
        and not args.allow_missing_prices
    ):
        raise RuntimeError(
            f"model catalog snapshot_date must be {today}, "
            f"got {catalog.get('snapshot_date')!r}"
        )

    routing_rows = []
    routing_runs: dict[str, Path] = {}
    for model in candidates:
        print(f"\n[phase 1] routing={model}, answer={BASELINE}")
        runs_file = runs_dir / f"routing__{model}.jsonl"
        if not (args.resume and runs_file.exists()):
            run_pipeline(
                dataset_path=args.dataset,
                profile="static-v1",
                out_path=runs_file,
                state_dir=args.state_dir,
                routing_model=model,
                answer_model=BASELINE,
                require_same_trace=True,
                workers=args.workers,
                sleep=args.sleep,
            )
        routing_runs[model] = runs_file

    review_template = output / "rewrite_review.json"
    if not review_template.exists():
        write_rewrite_review_template(review_template, routing_runs)
    reviews = (
        json.loads(args.rewrite_review.read_text(encoding="utf-8"))
        if args.rewrite_review
        else {}
    )
    for model in candidates:
        records = load_records(routing_runs[model])
        metrics = routing_metrics(records, reviews.get(model))
        routing_rows.append({
            "model": model,
            "metrics": metrics,
            "cost_per_1000": model_cost_per_1000(
                metrics["mean_input_tokens"],
                metrics["mean_output_tokens"],
                prices.get(model),
            ),
        })
    incomplete_reviews = [
        item["model"]
        for item in routing_rows
        if not item["metrics"]["human_review_complete"]
    ]
    if incomplete_reviews:
        print(
            "\nmanual rewrite review pending for: "
            + ", ".join(incomplete_reviews)
            + "; using deterministic rewrite metrics"
        )

    routing_rows.sort(key=lambda item: item["metrics"]["quality"], reverse=True)
    baseline_routing = next(
        item["metrics"] for item in routing_rows if item["model"] == BASELINE
    )
    eligible_routing = [
        item
        for item in routing_rows
        if item["metrics"]["intent_accuracy"]
        >= baseline_routing["intent_accuracy"] - 0.05
        and item["metrics"]["document_recall"]
        >= baseline_routing["document_recall"] - 0.03
        and item["metrics"]["stability"] >= 0.99
        and item["metrics"]["silent_condition_changes"]
        <= baseline_routing["silent_condition_changes"]
        and item["metrics"]["rewrite_metric_complete"]
    ]
    selected_routing = [
        item["model"]
        for item in (eligible_routing or routing_rows)[: args.top_routing]
    ]

    base_url = os.environ.get("RAGENT_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    username = os.environ.get("RAGENT_USERNAME")
    password = os.environ.get("RAGENT_PASSWORD")
    if not username or not password:
        raise RuntimeError("missing RAGENT_USERNAME / RAGENT_PASSWORD")
    token = login(base_url, username, password)

    arms: list[dict[str, Any]] = []
    for routing_model in selected_routing:
        source_file = routing_runs[routing_model]
        source_records = load_records(source_file)
        route = next(
            item["metrics"] for item in routing_rows if item["model"] == routing_model
        )
        for answer_model in candidates:
            print(f"\n[phase 2] routing={routing_model}, answer={answer_model}")
            if answer_model == BASELINE:
                arm_file = source_file
            else:
                arm_file = runs_dir / f"combo__{routing_model}__{answer_model}.jsonl"
                if not (args.resume and arm_file.exists()):
                    replay_answer_arm(
                        source_records=source_records,
                        routing_model=routing_model,
                        answer_model=answer_model,
                        base_url=base_url,
                        token=token,
                        out_path=arm_file,
                        dataset=args.dataset,
                    )
            if not (
                args.resume
                and score_is_reusable(
                    arm_file,
                    judge_model=judge_model,
                    judge_base_url=judge_base_url,
                    skip_ragas=args.skip_ragas,
                )
            ):
                score(
                    runs_file=arm_file,
                    skip_ragas=args.skip_ragas,
                    ragas_n=args.ragas_n,
                )
            answer = load_score_summary(arm_file)
            arms.append(
                {
                    "routing_model": routing_model,
                    "answer_model": answer_model,
                    "runs_file": str(arm_file),
                    "routing": route,
                    "answer": answer,
                    "answer_quality": answer_quality(answer),
                    "cost_per_1000": cost_per_1000(
                        route,
                        answer,
                        prices.get(routing_model),
                        prices.get(answer_model),
                    ),
                    "repeat_runs": [str(arm_file)],
                }
            )

    baseline = next(
        (
            arm
            for arm in arms
            if arm["routing_model"] == BASELINE
            and arm["answer_model"] == BASELINE
        ),
        None,
    )
    if baseline is None:
        baseline_file = routing_runs[BASELINE]
        if not (
            args.resume
            and score_is_reusable(
                baseline_file,
                judge_model=judge_model,
                judge_base_url=judge_base_url,
                skip_ragas=args.skip_ragas,
            )
        ):
            score(
                runs_file=baseline_file,
                skip_ragas=args.skip_ragas,
                ragas_n=args.ragas_n,
            )
        baseline_answer = load_score_summary(baseline_file)
        baseline = {
            "routing_model": BASELINE,
            "answer_model": BASELINE,
            "runs_file": str(baseline_file),
            "routing": baseline_routing,
            "answer": baseline_answer,
            "answer_quality": answer_quality(baseline_answer),
            "cost_per_1000": cost_per_1000(
                baseline_routing,
                baseline_answer,
                prices.get(BASELINE),
                prices.get(BASELINE),
            ),
            "repeat_runs": [str(baseline_file)],
        }
    ranked = final_rank(arms, baseline)

    finalists = ranked[: args.finalists]
    for arm in finalists:
        routing_runs_for_arm = [arm["routing"]]
        answer_runs_for_arm = [arm["answer"]]
        for repeat in range(2, args.final_rounds + 1):
            routing_model = arm["routing_model"]
            answer_model = arm["answer_model"]
            print(
                f"\n[final round {repeat}/{args.final_rounds}] "
                f"routing={routing_model}, answer={answer_model}"
            )
            repeat_file = (
                runs_dir
                / f"final__{routing_model}__{answer_model}__r{repeat}.jsonl"
            )
            run_pipeline(
                dataset_path=args.dataset,
                profile="static-v1",
                out_path=repeat_file,
                state_dir=args.state_dir,
                routing_model=routing_model,
                answer_model=answer_model,
                require_same_trace=True,
                workers=args.workers,
                sleep=args.sleep,
            )
            repeat_records = load_records(repeat_file)
            repeat_routing = routing_metrics(
                repeat_records,
                reviews.get(routing_model),
            )
            score(
                runs_file=repeat_file,
                skip_ragas=args.skip_ragas,
                ragas_n=args.ragas_n,
            )
            repeat_answer = load_score_summary(repeat_file)
            routing_runs_for_arm.append(repeat_routing)
            answer_runs_for_arm.append(repeat_answer)
            arm["repeat_runs"].append(str(repeat_file))
        arm["routing"] = average_metrics(routing_runs_for_arm)
        arm["answer"] = average_metrics(answer_runs_for_arm)
        arm["answer_quality"] = answer_quality(arm["answer"])
        arm["cost_per_1000"] = cost_per_1000(
            arm["routing"],
            arm["answer"],
            prices.get(arm["routing_model"]),
            prices.get(arm["answer_model"]),
        )

    ranked = final_rank(arms, baseline)
    eligible_arms = [arm for arm in ranked if arm["eligible"]]
    recommended = eligible_arms[0] if eligible_arms else None
    backup_candidates = [
        arm
        for arm in eligible_arms
        if recommended is None or arm is not recommended
    ]
    low_cost_backup = min(
        backup_candidates,
        key=lambda arm: (
            arm.get("cost_per_1000")
            if arm.get("cost_per_1000") is not None
            else float("inf"),
            arm["answer"].get("p95_ttft_ms")
            if arm["answer"].get("p95_ttft_ms") is not None
            else float("inf"),
        ),
        default=None,
    )
    result = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_sha256(args.dataset),
        "candidates": list(candidates),
        "judge_model": judge_model,
        "selected_routing_models": selected_routing,
        "routing_ranking": routing_rows,
        "combination_ranking": ranked,
        "recommendation": (
            {
                "routing_model": recommended["routing_model"],
                "answer_model": recommended["answer_model"],
                "final_score": recommended["final_score"],
            }
            if recommended
            else None
        ),
        "low_cost_backup": (
            {
                "routing_model": low_cost_backup["routing_model"],
                "answer_model": low_cost_backup["answer_model"],
                "cost_per_1000": low_cost_backup["cost_per_1000"],
            }
            if low_cost_backup
            else None
        ),
        "finalists": [
            {
                "routing_model": arm["routing_model"],
                "answer_model": arm["answer_model"],
                "repeat_runs": arm["repeat_runs"],
            }
            for arm in finalists
        ],
        "status_counts": dict(Counter(
            "eligible" if arm["eligible"] else "ineligible" for arm in ranked
        )),
        "model_catalog": catalog,
        "price_snapshot_required": bool(missing_prices),
        "human_rewrite_review_required": False,
        "hnsw_ef_search": 200,
        "retrieval_configuration_fixed": True,
        "rewrite_review_file": str(
            (args.rewrite_review or review_template).resolve()
        ),
        "pareto_svg": str((output / "pareto.svg").resolve()),
    }
    write_pareto_svg(output / "pareto.svg", ranked)
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "report.md").write_text(render_report(result), encoding="utf-8")
    print(f"\nresult: {output / 'result.json'}")
    print(f"report: {output / 'report.md'}")
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--state-dir", type=Path, default=None)
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--top-routing", type=int, default=2)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--final-rounds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--ragas-n", type=int, default=1)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--prices", type=Path, default=None)
    parser.add_argument("--allow-missing-prices", action="store_true")
    parser.add_argument("--rewrite-review", type=Path, default=None)
    parser.add_argument("--allow-pending-rewrite-review", action="store_true")
    parser.add_argument("--allow-pending-dataset-review", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
