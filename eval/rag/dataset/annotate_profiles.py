"""Apply the static-v1/tool-deferred annotations to evaluation datasets."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = PROJECT_ROOT / "eval" / "rag" / "dataset"
DATASET_PATHS = (
    DATASET_DIR / "eval_set_v1_all.jsonl",
    DATASET_DIR / "eval_set_v1.jsonl",
)

DEFERRED_ROUTES = {
    **{f"S1-{index:02d}": "TOOL" for index in range(1, 10)},
    **{
        f"S3-{index:02d}": "TOOL"
        for index in range(1, 10)
        if index != 8
    },
    "S4-01": "HYBRID",
    "S4-03": "TOOL",
    "S4-04": "TOOL",
    "S5-01": "HYBRID",
    "S5-03": "TOOL",
    "S5-05": "TOOL",
}


def _annotate(row: dict) -> dict:
    query_id = row["query_id"]
    route = DEFERRED_ROUTES.get(query_id)
    if route:
        row["expected_route"] = route
        row["evaluation_scope"] = "tool-deferred"
        row["scope_reason"] = (
            "依赖商品推荐、商品对比或动态业务数据，当前阶段暂缓评测。"
        )
        row["annotation_rationale"] = (
            "该问题需要工具执行结果才能形成可验证答案，不纳入静态 RAG 核心指标。"
        )
    else:
        route = "KB" if row.get("requires_rag", False) else "SYSTEM"
        row["expected_route"] = route
        row["evaluation_scope"] = "static-v1"
        row["scope_reason"] = "可由静态知识或系统边界话术完成。"
        row["annotation_rationale"] = (
            "该问题由静态商品、政策、手册或故障知识回答。"
            if route == "KB"
            else "该问题验证系统话术或边界处理，不应依赖知识库检索。"
        )

    row["requires_tool"] = route in {"TOOL", "HYBRID"}
    row["expected_doc_ids"] = [
        doc_id
        for doc_id in row.get("expected_doc_ids") or []
        if doc_id != "PRODUCT_MAPPING"
    ]
    row["expected_doc_ids_nice"] = [
        doc_id
        for doc_id in row.get("expected_doc_ids_nice") or []
        if doc_id != "PRODUCT_MAPPING"
    ]
    return row


def annotate_dataset(path: Path) -> None:
    """Rewrite one JSONL dataset with deterministic profile annotations."""
    rows = [
        _annotate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if path.name == "eval_set_v1_all.jsonl":
        deferred = [row for row in rows if row["evaluation_scope"] == "tool-deferred"]
        if len(deferred) != 23:
            raise RuntimeError(f"tool-deferred 应为 23 条，实际为 {len(deferred)} 条")
        if {row["query_id"] for row in deferred} != set(DEFERRED_ROUTES):
            raise RuntimeError("tool-deferred 样本集合与计划不一致")
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    for path in DATASET_PATHS:
        annotate_dataset(path)
        print(f"annotated {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
