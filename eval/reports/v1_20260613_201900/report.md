# 评测报告

> 数据源：`v1_20260613_201900.jsonl`
> 样本数：20
> 状态分布：{'success': 20}

> Profile：`static-v1`
> 纳入/排除：20 / 0
> 数据集 SHA-256：`ac1daf8be728f6a6c621b05a9f9fe36a76dbaaad9d8831dc2654fe78f1efe745`

## 自建指标

| 指标 | 数值 |
|---|---|
| 意图 Top-1 准确率 | 80.0% |
| Hit@1 | 0.0% |
| Hit@3 | 0.0% |
| Hit@5 | 0.0% |
| Hit@10 | 0.0% |
| Recall@5 | 0.0% |
| Recall@5 (must + nice) | 0.0% |
| Nice Hit@5 | 0.0% |
| Recall@10 | 0.0% |
| MRR@10 | 0.0% |
| 误拒率（requires_rag 却 0 召回） | 100.0% |
| 答案兜底率 | 95.0% |
| 过召回率（!requires_rag 却走 RAG） | — |
| SYSTEM Boundary Compliance | — |
| 首字延迟 P50 (ms) | 6180 |
| 首字延迟 P95 (ms) | 10137 |
| 首字延迟均值 (ms) | 6268 |
| 整流均值 (ms) | 6363 |

## 按 intent_l2 分层（核心指标）

| intent_l2 | Intent Top-1 | Hit@5 | Recall@5 | MRR@10 |
|---|---|---|---|---|
| F1_故障报告 | 100.0% | 0.0% | 0.0% | 0.0% |
| F3_投诉吐槽 | 100.0% | 0.0% | 0.0% | 0.0% |
| S10_APP功能 | 100.0% | 0.0% | 0.0% | 0.0% |
| S12_生态联动 | 100.0% | 0.0% | 0.0% | 0.0% |
| S13_保养维护 | 100.0% | 0.0% | 0.0% | 0.0% |
| S2_参数咨询 | 33.3% | 0.0% | 0.0% | 0.0% |
| S3_对比选购 | 100.0% | 0.0% | 0.0% | 0.0% |
| S4_价格活动 | 100.0% | 0.0% | 0.0% | 0.0% |
| S5_库存到货 | 100.0% | 0.0% | 0.0% | 0.0% |
| S6_配件兼容 | 100.0% | 0.0% | 0.0% | 0.0% |
| S7_适用场景 | 50.0% | 0.0% | 0.0% | 0.0% |
| S9_配网连接 | 66.7% | 0.0% | 0.0% | 0.0% |

## Retrieval Trace

| Stage | Count | P50 (ms) | P95 (ms) | Candidates avg/max |
|---|---:|---:|---:|---:|
| retrieval-scope-resolve | 19 | 0 | 1 | — |
| vector-intent-search | 19 | 708 | 5466 | 22.5/30 |
| keyword-pg-search | 19 | 3 | 17 | 1.5/15 |

> Slowest node: `S7-07` / `user-first-packet` / 10108ms

### Retrieval Bottlenecks (>2000ms)

| Query | Retrieval (ms) | Bottleneck stage | Stage (ms) |
|---|---:|---|---:|
| `S5-04` | 2749 | vector-intent-search | 2725 |
| `S10-03` | 2425 | vector-intent-search | 2403 |
| `S2-08` | 3088 | vector-intent-search | 3070 |
| `S7-05` | 2503 | vector-intent-search | 2483 |
| `S7-07` | 5485 | vector-intent-search | 5466 |

## 按难度分层（核心指标）

| difficulty | Intent Top-1 | Hit@5 | Recall@5 | MRR@10 |
|---|---|---|---|---|
| easy | 100.0% | 0.0% | 0.0% | 0.0% |
| medium | 85.7% | 0.0% | 0.0% | 0.0% |
| hard | 75.0% | 0.0% | 0.0% | 0.0% |
