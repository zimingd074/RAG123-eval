"""Isolated pgvector index validation for embedding benchmark vectors."""
from __future__ import annotations

import json
import subprocess
from typing import Any

import numpy as np

from eval.rag.embedding_benchmark import normalize

SCHEMA = "embedding_benchmark"


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(value):.8g}" for value in vector) + "]"


def _run_psql(
    *,
    container: str,
    database: str,
    user: str,
    sql: str,
) -> str:
    completed = subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            container,
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            user,
            "-d",
            database,
            "-At",
            "-F",
            "\t",
        ],
        input=sql,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return completed.stdout


def _python_exact_top10(
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
) -> list[list[int]]:
    scores = query_vectors @ document_vectors.T
    return [
        [int(index) for index in np.argsort(-row)[:10]]
        for row in scores
    ]


def _recall_at_10(expected: list[int], actual: list[int]) -> float:
    return len(set(expected).intersection(actual[:10])) / min(10, len(expected))


def validate_pgvector(
    *,
    container: str,
    database: str,
    user: str,
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    query_limit: int = 20,
    candidate_k: int = 50,
) -> dict[str, Any]:
    """Build one isolated HNSW index and compare it with float32 exact search."""
    if database.lower() in {"ragent", "postgres", "template0", "template1"}:
        raise ValueError(
            "Refusing pgvector validation in a shared/current database; "
            "provide a dedicated experiment database"
        )
    dimension = int(document_vectors.shape[1])
    if query_vectors.shape[1] != dimension:
        raise ValueError("Document and query vector dimensions do not match")
    documents = normalize(document_vectors.astype(np.float32, copy=False))
    queries = normalize(query_vectors.astype(np.float32, copy=False))[
        :query_limit
    ]
    storage_type = (
        f"vector({dimension})"
        if dimension <= 2000
        else f"halfvec({dimension})"
        if dimension <= 4000
        else f"vector({dimension})"
    )
    operator_class = (
        "vector_cosine_ops" if dimension <= 2000 else "halfvec_cosine_ops"
    )
    setup = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE;",
        f"CREATE SCHEMA {SCHEMA};",
    ]
    if dimension <= 4000:
        setup.append(
            f"CREATE TABLE {SCHEMA}.items "
            f"(id integer PRIMARY KEY, embedding {storage_type} NOT NULL);"
        )
        values = ",\n".join(
            f"({index}, '{_vector_literal(vector)}')"
            for index, vector in enumerate(documents)
        )
        setup.extend(
            [
                f"INSERT INTO {SCHEMA}.items (id, embedding) VALUES {values};",
                f"CREATE INDEX embedding_hnsw ON {SCHEMA}.items "
                f"USING hnsw (embedding {operator_class});",
                f"ANALYZE {SCHEMA}.items;",
            ]
        )
        _run_psql(
            container=container,
            database=database,
            user=user,
            sql="\n".join(setup),
        )
        query_sql = ["SET enable_seqscan=off;", "SET hnsw.ef_search=100;"]
        for query_index, vector in enumerate(queries):
            literal = _vector_literal(vector)
            query_sql.append(
                "SELECT "
                f"{query_index}, array_to_json(array_agg(id ORDER BY distance)) "
                "FROM (SELECT id, embedding <=> "
                f"'{literal}'::{storage_type} AS distance "
                f"FROM {SCHEMA}.items ORDER BY embedding <=> "
                f"'{literal}'::{storage_type} LIMIT 10) ranked;"
            )
        output = _run_psql(
            container=container,
            database=database,
            user=user,
            sql="\n".join(query_sql),
        )
        actual = {
            int(parts[0]): [int(value) for value in json.loads(parts[1])]
            for line in output.splitlines()
            if len(parts := line.split("\t", 1)) == 2
        }
        exact = _python_exact_top10(documents, queries)
        recalls = [
            _recall_at_10(exact[index], actual.get(index, []))
            for index in range(len(queries))
        ]
        half_precision_cosine = None
        if dimension > 2000:
            quantized = normalize(
                documents.astype(np.float16).astype(np.float32)
            )
            half_precision_cosine = {
                "mean": float(np.sum(documents * quantized, axis=1).mean()),
                "min": float(np.sum(documents * quantized, axis=1).min()),
            }
        index_count_sql = (
            "SELECT count(*) FROM pg_indexes "
            f"WHERE schemaname='{SCHEMA}' AND indexdef ILIKE '%USING hnsw%';"
        )
        index_count = int(
            _run_psql(
                container=container,
                database=database,
                user=user,
                sql=index_count_sql,
            ).strip()
        )
        return {
            "dimension": dimension,
            "storage": storage_type,
            "query_count": len(queries),
            "ann_recall@10_mean": float(statistics_mean(recalls)),
            "ann_recall@10_min": float(min(recalls)),
            "hnsw_index_count": index_count,
            "single_hnsw_index": index_count == 1,
            "half_precision_cosine": half_precision_cosine,
        }

    prefix_dimension = 1536
    setup.append(
        f"CREATE TABLE {SCHEMA}.items ("
        "id integer PRIMARY KEY, "
        f"embedding vector({dimension}) NOT NULL, "
        f"prefix vector({prefix_dimension}) NOT NULL);"
    )
    values = ",\n".join(
        f"({index}, '{_vector_literal(vector)}', "
        f"'{_vector_literal(normalize(vector[:prefix_dimension][None, :])[0])}')"
        for index, vector in enumerate(documents)
    )
    setup.extend(
        [
            f"INSERT INTO {SCHEMA}.items (id, embedding, prefix) VALUES {values};",
            f"CREATE INDEX embedding_prefix_hnsw ON {SCHEMA}.items "
            "USING hnsw (prefix vector_cosine_ops);",
            f"ANALYZE {SCHEMA}.items;",
        ]
    )
    _run_psql(
        container=container,
        database=database,
        user=user,
        sql="\n".join(setup),
    )
    exact = _python_exact_top10(documents, queries)
    candidate_recalls: list[float] = []
    final_recalls: list[float] = []
    for query_index, vector in enumerate(queries):
        prefix = normalize(vector[:prefix_dimension][None, :])[0]
        sql = (
            "SET enable_seqscan=off; SET hnsw.ef_search=100; "
            "SELECT id FROM "
            f"{SCHEMA}.items ORDER BY prefix <=> "
            f"'{_vector_literal(prefix)}'::vector({prefix_dimension}) "
            f"LIMIT {candidate_k};"
        )
        candidates = [
            int(line)
            for line in _run_psql(
                container=container,
                database=database,
                user=user,
                sql=sql,
            ).splitlines()
            if line.strip()
        ]
        candidate_recalls.append(
            len(set(exact[query_index]).intersection(candidates)) / 10
        )
        reranked = sorted(
            candidates,
            key=lambda index: float(documents[index] @ vector),
            reverse=True,
        )[:10]
        final_recalls.append(_recall_at_10(exact[query_index], reranked))
    return {
        "dimension": dimension,
        "storage": storage_type,
        "query_count": len(queries),
        "strategy": "1536-dimensional HNSW candidates plus 4096-dimensional rerank",
        "candidate_k": candidate_k,
        "candidate_recall@10_mean": float(statistics_mean(candidate_recalls)),
        "rerank_recall@10_mean": float(statistics_mean(final_recalls)),
        "hnsw_index_count": 1,
        "single_hnsw_index": True,
    }


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
