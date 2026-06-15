"""Query rewrite fidelity metrics.

The rewrite stage must improve retrieval wording without silently changing the
user's constraints. These metrics are deterministic and can be recomputed from
the recorded ``eval-query-rewrite`` trace without another model call.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from eval.common.schemas import EvalRecord, MetricResult
from eval.rag.metrics._common import slice_mean


_ENTITY_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Za-z]+(?:[\s._/+:-]*[A-Za-z0-9]+)+|[A-Za-z]+\d+|\d+[A-Za-z]+)"
    r"(?![A-Za-z0-9])"
)
_QUANTITY_RE = re.compile(
    r"([零〇一二两三四五六七八九十百千万\d.]+)\s*"
    r"(天|周|星期|个月|月|年|小时|分钟|元|块|瓦|w|mah|gb|tb|种)",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"不|没|未|无|无法|不能|不可|禁止|别")
_COMPLAINT_RE = re.compile(r"太慢|投诉|不满|生气|差评|离谱|糟糕|坑人")

_SCOPE_TERMS = (
    "只",
    "仅",
    "至少",
    "最多",
    "以内",
    "以上",
    "以下",
    "低成本",
    "带走",
    "自营",
    "原装",
    "无线",
    "有线",
)
_COMPARISON_TERMS = ("更", "最", "比", "优先", "便宜", "贵", "高于", "低于")
_QUESTION_PATTERNS = {
    "price": re.compile(r"多少钱|价格|费用|收费"),
    "count": re.compile(r"几种|多少种|几个|数量"),
    "when": re.compile(r"什么时候|何时|多久|上市时间|发货时间"),
    "who": re.compile(r"谁|哪方|由谁"),
}
_PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)


@dataclass(frozen=True)
class RewriteEvaluation:
    """Per-sample rewrite analysis."""

    original: str
    rewritten: str
    semantic_score: float | None
    condition_recall: float | None
    condition_precision: float | None
    silent_condition_change: bool | None
    missing_conditions: tuple[str, ...] = ()
    added_conditions: tuple[str, ...] = ()


def _trace_node_extra(
    trace_detail: dict[str, Any] | None,
    node_name: str,
) -> dict[str, Any]:
    if not trace_detail:
        return {}
    for node in reversed(trace_detail.get("nodes") or []):
        if node.get("nodeName") == node_name:
            extra = node.get("extraData")
            return extra if isinstance(extra, dict) else {}
    return {}


def _rewrite_pair(record: EvalRecord) -> tuple[str, str]:
    snapshot = _trace_node_extra(
        getattr(record, "chat_trace", None),
        "eval-query-rewrite",
    )
    original = str(
        snapshot.get("originalQuestion")
        or getattr(record, "user_input", "")
        or ""
    ).strip()
    rewritten = str(snapshot.get("rewrittenQuestion") or "").strip()
    return original, rewritten


def _chinese_number(value: str) -> str:
    if value.replace(".", "", 1).isdigit():
        return value
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = 0
    section = 0
    number = 0
    for char in value:
        if char in digits:
            number = digits[char]
        elif char in units:
            unit = units[char]
            if unit == 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                section += (number or 1) * unit
            number = 0
        else:
            return value
    return str(total + section + number)


def extract_conditions(text: str) -> set[str]:
    """Extract facts that a rewrite is not allowed to add, remove, or alter."""
    normalized = unicodedata.normalize("NFKC", text or "")
    lowered = normalized.lower()
    conditions: set[str] = set()

    for match in _ENTITY_RE.finditer(normalized):
        entity = re.sub(r"[\s._/+:-]+", "", match.group(0)).lower()
        if entity not in {"app"}:
            conditions.add(f"entity:{entity}")
        elif "app" in lowered:
            conditions.add("entity:app")

    for value, unit in _QUANTITY_RE.findall(lowered):
        canonical_unit = {"星期": "周", "块": "元", "w": "瓦"}.get(unit.lower(), unit.lower())
        conditions.add(f"quantity:{_chinese_number(value)}{canonical_unit}")

    if _NEGATION_RE.search(normalized):
        conditions.add("polarity:negative")
    if _COMPLAINT_RE.search(normalized):
        conditions.add("tone:complaint")

    for term in _SCOPE_TERMS:
        if term in normalized:
            conditions.add(f"scope:{term}")
    for term in _COMPARISON_TERMS:
        if term in normalized:
            conditions.add(f"comparison:{term}")
    for name, pattern in _QUESTION_PATTERNS.items():
        if pattern.search(normalized):
            conditions.add(f"question:{name}")
    return conditions


def _canonical_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    replacements = {
        "能不能": "能否",
        "是否可以": "能否",
        "可不可以": "能否",
        "怎么样": "如何",
        "怎么": "如何",
        "一星期": "一周",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return _PUNCT_RE.sub("", normalized)


def _char_bigram_f1(original: str, rewritten: str) -> float:
    left = _canonical_text(original)
    right = _canonical_text(rewritten)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    def grams(value: str) -> set[str]:
        if len(value) < 2:
            return {value}
        return {value[index : index + 2] for index in range(len(value) - 1)}

    left_grams = grams(left)
    right_grams = grams(right)
    overlap = len(left_grams & right_grams)
    precision = overlap / len(right_grams)
    recall = overlap / len(left_grams)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def evaluate_record(record: EvalRecord) -> RewriteEvaluation:
    """Evaluate one recorded query rewrite."""
    original, rewritten = _rewrite_pair(record)
    if not original or not rewritten:
        return RewriteEvaluation(original, rewritten, None, None, None, None)

    original_conditions = extract_conditions(original)
    rewritten_conditions = extract_conditions(rewritten)
    missing = original_conditions - rewritten_conditions
    added = rewritten_conditions - original_conditions
    condition_recall = (
        len(original_conditions & rewritten_conditions) / len(original_conditions)
        if original_conditions
        else 1.0
    )
    condition_precision = (
        len(original_conditions & rewritten_conditions) / len(rewritten_conditions)
        if rewritten_conditions
        else (1.0 if not original_conditions else 0.0)
    )
    condition_f1 = (
        2
        * condition_precision
        * condition_recall
        / (condition_precision + condition_recall)
        if condition_precision + condition_recall
        else 0.0
    )
    lexical_f1 = _char_bigram_f1(original, rewritten)
    semantic_score = 0.70 * condition_f1 + 0.30 * lexical_f1
    return RewriteEvaluation(
        original=original,
        rewritten=rewritten,
        semantic_score=semantic_score,
        condition_recall=condition_recall,
        condition_precision=condition_precision,
        silent_condition_change=bool(missing or added),
        missing_conditions=tuple(sorted(missing)),
        added_conditions=tuple(sorted(added)),
    )


def compute(records: list[EvalRecord]) -> list[MetricResult]:
    """Return rewrite parse, fidelity, and silent-change metrics."""
    evaluations = {record.query_id: evaluate_record(record) for record in records}

    def semantic(record: EvalRecord) -> float | None:
        return evaluations[record.query_id].semantic_score

    def condition_recall(record: EvalRecord) -> float | None:
        return evaluations[record.query_id].condition_recall

    def silent_change(record: EvalRecord) -> float | None:
        value = evaluations[record.query_id].silent_condition_change
        return None if value is None else float(value)

    def parsed(record: EvalRecord) -> float:
        return float(bool(evaluations[record.query_id].rewritten))

    results = []
    for name, value_fn in (
        ("rewrite_parse_rate", parsed),
        ("rewrite_semantic_preservation", semantic),
        ("rewrite_condition_recall", condition_recall),
        ("silent_condition_change_rate", silent_change),
    ):
        overall, by_l1, by_l2, per_sample = slice_mean(
            records,
            value_fn,
            lambda record: getattr(record, "evaluation_scope", "static-v1")
            != "tool-deferred",
        )
        meta: dict[str, Any] = {
            "method": (
                "protected-condition precision/recall plus normalized character "
                "bigram F1; no model call"
            ),
            "evaluated_count": sum(
                evaluations[record.query_id].semantic_score is not None
                for record in records
                if getattr(record, "evaluation_scope", "static-v1")
                != "tool-deferred"
            ),
        }
        if name == "silent_condition_change_rate":
            meta["changed_count"] = sum(value == 1.0 for value in per_sample.values())
            meta["changes"] = {
                query_id: {
                    "missing": list(evaluation.missing_conditions),
                    "added": list(evaluation.added_conditions),
                }
                for query_id, evaluation in evaluations.items()
                if evaluation.silent_condition_change
            }
        results.append(
            MetricResult(
                name=name,
                overall=overall,
                by_intent_l1=by_l1,
                by_intent_l2=by_l2,
                per_sample=per_sample,
                meta=meta,
            )
        )
    return results
