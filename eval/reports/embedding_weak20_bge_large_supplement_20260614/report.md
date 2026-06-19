# Embedding 模型选型报告

- 样本数：20
- Chunk 数：141
- 主判定作用域：gold_collection
- 推荐配置：无
- 判定说明：No arm passed all configured quality gates.

## 质量与工程指标

| Arm | 维度 | Hit@5 | Recall@5 | MRR@10 | Query P95(ms) | 每万次查询成本 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| bge-large-zh-1024 | 1024 | 1.000 | 0.762 | 0.758 | 139.4 | n/a | FAIL |

## 说明

- 质量检索为 chunk 级精确余弦排序，再按文档 ID 去重；未启用关键词、RRF 或 rerank。
- `global` 为全库检索，`gold_collection` 仅在 gold 文档所属知识库中检索。
- BGE 请求不发送 `dimensions`；Qwen 请求显式发送目标维度。
- 2048 维以上的索引方案与精度风险见 `matrix.json` 的 `storage_plan`。
