# Static Weak-20 A/B Comparison

- Baseline: `v1_20260612_164916`, the latest pre-change 127-sample static run.
- Candidate: `v1_20260613_012555`, rerun on the fixed 20 weak samples only.
- Selection: `requires_rag=true`, ranked by the mean of available intent, Hit@5,
  Recall@5, MRR@10, and five RAGAS metrics.
- Baseline RAGAS values are reused from the original full-run per-sample scores.

## Overall Metrics

| Metric | Baseline | Candidate | Delta |
|---|---:|---:|---:|
| intent_top1 | 80.0% | 85.0% | +5.0 pp |
| hit@5 | 90.0% | 95.0% | +5.0 pp |
| recall@5 | 59.2% | 65.0% | +5.8 pp |
| mrr@10 | 69.2% | 72.9% | +3.8 pp |
| faithfulness | 66.0% | 60.5% | -5.5 pp |
| answer_relevancy | 43.4% | 43.3% | -0.1 pp |
| answer_correctness | 47.9% | 52.4% | +4.6 pp |
| context_precision | 73.5% | 72.1% | -1.4 pp |
| context_recall | 25.4% | 28.3% | +2.9 pp |
| ttft_p50_ms | 6246 ms | 6647 ms | +401 ms |
| ttft_p95_ms | 8456 ms | 12793 ms | +4337 ms |
| total_mean_ms | 9708 ms | 10519 ms | +810 ms |

## Per-Sample Result

- Improved: 11
- Stable (absolute delta <= 0.01): 2
- Regressed: 7

| query_id | Intent | Baseline | Candidate | Delta |
|---|---|---:|---:|---:|
| F3-03 | F3_投诉吐槽 | 0.250 | 0.704 | +0.454 |
| F1-05 | F1_故障报告 | 0.569 | 0.674 | +0.105 |
| S12-03 | S12_生态联动 | 0.696 | 0.797 | +0.100 |
| S7-05 | S7_适用场景 | 0.603 | 0.685 | +0.082 |
| S7-07 | S7_适用场景 | 0.664 | 0.739 | +0.075 |
| S2-09 | S2_参数咨询 | 0.538 | 0.588 | +0.050 |
| S5-04 | S5_库存到货 | 0.319 | 0.368 | +0.050 |
| S10-03 | S10_APP功能 | 0.584 | 0.626 | +0.042 |
| S9-07 | S9_配网连接 | 0.575 | 0.611 | +0.036 |
| S3-08 | S3_对比选购 | 0.706 | 0.728 | +0.023 |
| S9-02 | S9_配网连接 | 0.694 | 0.707 | +0.013 |
| S13-09 | S13_保养维护 | 0.651 | 0.655 | +0.004 |
| S6-04 | S6_配件兼容 | 0.710 | 0.703 | -0.007 |
| S2-08 | S2_参数咨询 | 0.597 | 0.583 | -0.014 |
| S4-05 | S4_价格活动 | 0.656 | 0.619 | -0.036 |
| S9-05 | S9_配网连接 | 0.681 | 0.624 | -0.056 |
| S2-11 | S2_参数咨询 | 0.690 | 0.627 | -0.063 |
| S12-04 | S12_生态联动 | 0.685 | 0.621 | -0.065 |
| S12-05 | S12_生态联动 | 0.684 | 0.611 | -0.073 |
| S12-06 | S12_生态联动 | 0.625 | 0.497 | -0.128 |

## Interpretation

- Retrieval improved across Hit@5, Recall@5, MRR@10, and context recall.
- Answer correctness improved, while faithfulness regressed by 5.5 percentage points.
- P95 time to first token regressed by 4.34 seconds and needs investigation.
- The ragent worktree also changes intent classification and answer prompts, so answer
  metric changes cannot be attributed to multi-channel retrieval alone.
- RAGAS is stochastic; retrieval metrics and latency are the stronger signals here.
