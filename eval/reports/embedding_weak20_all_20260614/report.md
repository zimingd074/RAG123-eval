# Embedding 模型选型报告

- 样本数：20
- Chunk 数：141
- 主判定作用域：gold_collection
- 推荐配置：无
- 判定说明：No arm passed all configured quality gates.

## 质量与工程指标

| Arm | 维度 | Hit@5 | Recall@5 | MRR@10 | Query P95(ms) | 每万次查询成本 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| qwen3-8b-1536-instructed | 1536 | 0.950 | 0.742 | 0.825 | 6640.5 | n/a | FAIL |
| qwen3-8b-1536 | 1536 | 0.950 | 0.742 | 0.825 | 4804.8 | n/a | FAIL |
| qwen3-0.6b-1024 | 1024 | 1.000 | 0.767 | 0.867 | 103.3 | n/a | FAIL |
| qwen3-8b-2560 | 2560 | 0.950 | 0.725 | 0.825 | 4779.3 | n/a | FAIL |
| qwen3-8b-512 | 512 | 1.000 | 0.767 | 0.793 | 3675.0 | n/a | FAIL |
| bge-m3-1024 | 1024 | 1.000 | 0.779 | 0.750 | 429.8 | n/a | FAIL |
| qwen3-4b-1024 | 1024 | 1.000 | 0.792 | 0.833 | 451.2 | n/a | FAIL |
| qwen3-8b-1536-current | 1536 | 0.950 | 0.713 | 0.771 | 3883.5 | n/a | FAIL |
| qwen3-8b-4096 | 4096 | 0.950 | 0.725 | 0.825 | 4285.2 | n/a | FAIL |
| qwen3-8b-2048 | 2048 | 0.950 | 0.742 | 0.825 | 5066.3 | n/a | FAIL |
| qwen3-8b-1024 | 1024 | 0.950 | 0.758 | 0.825 | 3552.8 | n/a | FAIL |
| bge-large-zh-1024 | 1024 | 1.000 | 0.762 | 0.758 | 139.4 | n/a | FAIL |

## 说明

- 质量检索为 chunk 级精确余弦排序，再按文档 ID 去重；未启用关键词、RRF 或 rerank。
- `global` 为全库检索，`gold_collection` 仅在 gold 文档所属知识库中检索。
- BGE 请求不发送 `dimensions`；Qwen 请求显式发送目标维度。
- 2048 维以上的索引方案与精度风险见 `matrix.json` 的 `storage_plan`。

## MRL 探针

- 4096 前 1536 维归一化 vs API 原生 1536 平均余弦：0.99991477
- 最低余弦：0.99988401
- 最大绝对差：0.00193646

## BGE-M3 服务探针

- 普通版 vs Pro 最低余弦：0.99996042
- 向量等价：是
- 处理方式：Reuse BAAI/bge-m3 quality results; compare only latency/stability.
