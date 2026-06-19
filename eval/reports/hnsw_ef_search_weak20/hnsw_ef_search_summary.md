# HNSW ef_search Weak-20 Sweep

Dataset: `D:\PycharmProjects\ragenteval\eval\rag\dataset\eval_set_static_weak20_20260613_groundtruth_fixed.jsonl`

| ef_search | round | hit@5 | recall@5 | mrr@10 | vector_search_p95_ms | retrieval_p95_ms | wall_p95_ms | missed_cases |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 40 | 1 | 0.95 | 0.616667 | 0.668333 | 44 | 2159 | 2426.4 | F3-03;S5-04;F1-05;S9-07;S7-05;S12-06;S13-09;S4-05;S7-07;S9-05;S12-05;S12-04;S9-02;S12-03 |
| 120 | 1 | 0.95 | 0.616667 | 0.668333 | 59 | 2212 | 2481.9 | F3-03;S5-04;F1-05;S9-07;S7-05;S12-06;S13-09;S4-05;S7-07;S9-05;S12-05;S12-04;S9-02;S12-03 |
| 200 | 1 | 0.95 | 0.616667 | 0.668333 | 47 | 1343 | 1636.1 | F3-03;S5-04;F1-05;S9-07;S7-05;S12-06;S13-09;S4-05;S7-07;S9-05;S12-05;S12-04;S9-02;S12-03 |
| 400 | 1 | 0.95 | 0.616667 | 0.668333 | 65 | 1342 | 1619.83 | F3-03;S5-04;F1-05;S9-07;S7-05;S12-06;S13-09;S4-05;S7-07;S9-05;S12-05;S12-04;S9-02;S12-03 |

## Decision

Keep 200: it is close to the 400 recall ceiling without the high-ef latency tradeoff.

CSV: `D:\PycharmProjects\ragenteval\eval\reports\hnsw_ef_search_weak20\hnsw_ef_search_summary.csv`
