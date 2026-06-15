from __future__ import annotations

import unittest

import numpy as np

from eval.rag.embedding_benchmark import (
    BGE_ZH_QUERY_PREFIX,
    QWEN_QUERY_PREFIX,
    EmbeddingArm,
    embedding_request_body,
    rank_documents,
    retrieval_metrics,
    storage_plan,
)
from eval.rag.pgvector_validation import validate_pgvector


class EmbeddingBenchmarkTest(unittest.TestCase):
    def test_qwen_query_uses_instruction_and_dimensions(self) -> None:
        arm = EmbeddingArm(
            "qwen",
            "Qwen/Qwen3-Embedding-8B",
            1536,
            True,
            QWEN_QUERY_PREFIX,
        )

        query_body = embedding_request_body(arm, ["怎么退货"], query=True)
        document_body = embedding_request_body(arm, ["退货政策"], query=False)

        self.assertEqual(query_body["dimensions"], 1536)
        self.assertEqual(query_body["input"], [f"{QWEN_QUERY_PREFIX}怎么退货"])
        self.assertEqual(document_body["input"], ["退货政策"])

    def test_bge_does_not_send_dimensions(self) -> None:
        arm = EmbeddingArm(
            "bge",
            "BAAI/bge-large-zh-v1.5",
            1024,
            False,
            BGE_ZH_QUERY_PREFIX,
        )

        body = embedding_request_body(arm, ["怎么退货"], query=True)

        self.assertNotIn("dimensions", body)
        self.assertEqual(body["input"], [f"{BGE_ZH_QUERY_PREFIX}怎么退货"])

    def test_bge_large_zh_truncates_long_inputs(self) -> None:
        arm = EmbeddingArm(
            "bge",
            "BAAI/bge-large-zh-v1.5",
            1024,
            False,
            BGE_ZH_QUERY_PREFIX,
            max_input_chars=512,
        )

        body = embedding_request_body(arm, ["文" * 800], query=False)

        self.assertEqual(len(body["input"][0]), 512)

    def test_retrieval_metrics_cover_required_cutoffs(self) -> None:
        metrics = retrieval_metrics(["A", "B"], ["X", "A", "Y", "B"])

        self.assertEqual(metrics["hit@1"], 0)
        self.assertEqual(metrics["hit@3"], 1)
        self.assertEqual(metrics["recall@5"], 1)
        self.assertAlmostEqual(metrics["mrr@10"], 0.5)
        self.assertGreater(metrics["ndcg@10"], 0)

    def test_document_ranking_deduplicates_chunks(self) -> None:
        from eval.rag.embedding_benchmark import CorpusChunk

        corpus = [
            CorpusChunk("c1", "A", "ra", "product", "ka", "one"),
            CorpusChunk("c2", "A", "ra", "product", "ka", "two"),
            CorpusChunk("c3", "B", "rb", "manual", "kb", "three"),
        ]
        documents = np.asarray([[1.0, 0], [0.9, 0.1], [0.8, 0.2]])
        documents /= np.linalg.norm(documents, axis=1, keepdims=True)

        ranked = rank_documents(
            np.asarray([1.0, 0]),
            documents,
            corpus,
            allowed_kbs=None,
        )

        self.assertEqual(ranked, ["A", "B"])

    def test_storage_plan_respects_pgvector_hnsw_limits(self) -> None:
        self.assertEqual(storage_plan(1536)["storage"], "vector(1536)")
        self.assertTrue(storage_plan(1536)["hnsw"])
        self.assertEqual(storage_plan(2560)["storage"], "halfvec(2560)")
        self.assertTrue(storage_plan(2560)["hnsw"])
        self.assertFalse(storage_plan(4096)["hnsw"])

    def test_pgvector_validation_rejects_current_database(self) -> None:
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        with self.assertRaisesRegex(
            ValueError, "dedicated experiment database"
        ):
            validate_pgvector(
                container="postgres",
                database="ragent",
                user="postgres",
                document_vectors=vectors,
                query_vectors=vectors,
            )


if __name__ == "__main__":
    unittest.main()
