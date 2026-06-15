"""Embedding model benchmark for the fixed ragent knowledge corpus."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests

from eval.common.env import load_project_env
from eval.common.schemas import EvalSample, load_samples
from eval.rag.pipeline.runner import login

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = (
    PROJECT_ROOT / "eval" / "rag" / "dataset" / "eval_set_v1_all.jsonl"
)
DEFAULT_DOC_MAP = (
    PROJECT_ROOT / "eval" / "rag" / "dataset" / "doc_id_map.json"
)
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
DEFAULT_BASE_URL = "http://localhost:9090/api/ragent"
DEFAULT_SILICONFLOW_URL = "https://api.siliconflow.cn/v1"

QWEN_QUERY_PREFIX = (
    "Instruct: Given a user question, retrieve relevant passages from a "
    "Chinese e-commerce knowledge base that answer the question.\nQuery: "
)
BGE_ZH_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


@dataclass(frozen=True)
class EmbeddingArm:
    arm_id: str
    model: str
    dimension: int
    send_dimensions: bool
    query_prefix: str = ""
    family: str = ""
    note: str = ""
    max_input_chars: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EmbeddingArm":
        return cls(
            arm_id=str(payload["arm_id"]),
            model=str(payload["model"]),
            dimension=int(payload["dimension"]),
            send_dimensions=bool(payload.get("send_dimensions", True)),
            query_prefix=str(payload.get("query_prefix") or ""),
            family=str(payload.get("family") or ""),
            note=str(payload.get("note") or ""),
            max_input_chars=(
                int(payload["max_input_chars"])
                if payload.get("max_input_chars") is not None
                else None
            ),
        )


@dataclass
class ApiTelemetry:
    latencies_ms: list[float] = field(default_factory=list)
    prompt_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0
    error_count: int = 0
    cache_hit: bool = False

    def merge(self, other: "ApiTelemetry") -> None:
        self.latencies_ms.extend(other.latencies_ms)
        self.prompt_tokens += other.prompt_tokens
        self.total_tokens += other.total_tokens
        self.request_count += other.request_count
        self.error_count += other.error_count
        self.cache_hit = self.cache_hit or other.cache_hit


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    business_doc_id: str
    ragent_doc_id: str
    kb_key: str
    kb_id: str
    content: str


def semantic_arms() -> list[EmbeddingArm]:
    return [
        EmbeddingArm(
            "qwen3-0.6b-1024",
            "Qwen/Qwen3-Embedding-0.6B",
            1024,
            True,
            QWEN_QUERY_PREFIX,
            "qwen3",
        ),
        EmbeddingArm(
            "qwen3-4b-1024",
            "Qwen/Qwen3-Embedding-4B",
            1024,
            True,
            QWEN_QUERY_PREFIX,
            "qwen3",
        ),
        EmbeddingArm(
            "qwen3-8b-1024",
            "Qwen/Qwen3-Embedding-8B",
            1024,
            True,
            QWEN_QUERY_PREFIX,
            "qwen3",
        ),
        EmbeddingArm(
            "bge-m3-1024",
            "BAAI/bge-m3",
            1024,
            False,
            "",
            "bge",
        ),
        EmbeddingArm(
            "bge-large-zh-1024",
            "BAAI/bge-large-zh-v1.5",
            1024,
            False,
            BGE_ZH_QUERY_PREFIX,
            "bge",
            "SiliconFlow rejects long project chunks for this model; "
            "documents are truncated conservatively.",
            512,
        ),
        EmbeddingArm(
            "qwen3-8b-1536-current",
            "Qwen/Qwen3-Embedding-8B",
            1536,
            True,
            "",
            "qwen3",
            "Current project configuration without query instruction.",
        ),
        EmbeddingArm(
            "qwen3-8b-1536-instructed",
            "Qwen/Qwen3-Embedding-8B",
            1536,
            True,
            QWEN_QUERY_PREFIX,
            "qwen3",
            "Current dimension with official query instruction format.",
        ),
    ]


def dimension_arms() -> list[EmbeddingArm]:
    dimensions = (512, 1024, 1536, 2048, 2560, 4096)
    return [
        EmbeddingArm(
            f"qwen3-8b-{dimension}",
            "Qwen/Qwen3-Embedding-8B",
            dimension,
            True,
            QWEN_QUERY_PREFIX,
            "qwen3",
        )
        for dimension in dimensions
    ]


def load_arms(preset: str, matrix_path: Path | None) -> list[EmbeddingArm]:
    if matrix_path:
        payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        rows = payload.get("arms", payload) if isinstance(payload, dict) else payload
        return [EmbeddingArm.from_dict(row) for row in rows]
    if preset == "semantic":
        return semantic_arms()
    if preset == "dimensions":
        return dimension_arms()
    by_id = {arm.arm_id: arm for arm in semantic_arms() + dimension_arms()}
    return list(by_id.values())


def embedding_request_body(
    arm: EmbeddingArm,
    texts: list[str],
    *,
    query: bool,
) -> dict[str, Any]:
    inputs = []
    for text in texts:
        prepared = (
            text[: arm.max_input_chars]
            if arm.max_input_chars is not None
            else text
        )
        if query and arm.query_prefix:
            prepared = f"{arm.query_prefix}{prepared}"
        inputs.append(prepared)
    body: dict[str, Any] = {
        "model": arm.model,
        "input": inputs,
        "encoding_format": "float",
    }
    if arm.send_dimensions:
        body["dimensions"] = arm.dimension
    return body


class SiliconFlowEmbeddingClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout: float,
        retries: int,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/embeddings"
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def embed(
        self,
        arm: EmbeddingArm,
        texts: list[str],
        *,
        query: bool,
    ) -> tuple[np.ndarray, ApiTelemetry]:
        telemetry = ApiTelemetry()
        body = embedding_request_body(arm, texts, query=query)
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            started = time.perf_counter()
            telemetry.request_count += 1
            try:
                response = self.session.post(
                    self.url,
                    json=body,
                    timeout=self.timeout,
                )
                telemetry.latencies_ms.append(
                    (time.perf_counter() - started) * 1000
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                response.raise_for_status()
                payload = response.json()
                ordered = sorted(
                    payload.get("data") or [],
                    key=lambda row: int(row.get("index", 0)),
                )
                vectors = np.asarray(
                    [row["embedding"] for row in ordered],
                    dtype=np.float32,
                )
                if vectors.shape != (len(texts), arm.dimension):
                    raise ValueError(
                        f"{arm.arm_id} returned shape {vectors.shape}, "
                        f"expected ({len(texts)}, {arm.dimension})"
                    )
                usage = payload.get("usage") or {}
                telemetry.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                telemetry.total_tokens += int(usage.get("total_tokens") or 0)
                return normalize(vectors), telemetry
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                telemetry.error_count += 1
                if attempt >= self.retries:
                    break
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise RuntimeError(
            f"Embedding request failed after {self.retries + 1} attempts: "
            f"{last_error}"
        )


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Embedding API returned a zero vector")
    return vectors / norms


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def embed_many(
    client: SiliconFlowEmbeddingClient,
    arm: EmbeddingArm,
    texts: list[str],
    *,
    query: bool,
    batch_size: int,
) -> tuple[np.ndarray, ApiTelemetry]:
    all_vectors: list[np.ndarray] = []
    telemetry = ApiTelemetry()
    effective_batch = 1 if query else batch_size
    for batch in chunks(texts, effective_batch):
        vectors, request_telemetry = client.embed(arm, batch, query=query)
        all_vectors.append(vectors)
        telemetry.merge(request_telemetry)
    return np.vstack(all_vectors), telemetry


def _payload_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data") or {}
    if isinstance(data, list):
        return data
    return list(data.get("records") or data.get("list") or [])


def fetch_corpus(
    *,
    base_url: str,
    username: str,
    password: str,
    doc_map_path: Path,
    include_unmapped: bool = True,
) -> list[CorpusChunk]:
    doc_map = json.loads(doc_map_path.read_text(encoding="utf-8"))
    token = login(base_url, username, password)
    headers = {"Authorization": token, "Accept": "application/json"}
    corpus: list[CorpusChunk] = []
    seen: set[str] = set()

    def append_document(
        business_doc_id: str,
        ragent_doc_id: str,
        kb_key: str,
        kb_id: str,
    ) -> None:
        response = requests.get(
            f"{base_url}/knowledge-base/docs/{ragent_doc_id}/chunks",
            params={"current": 1, "size": 500, "enabled": 1},
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success", True):
            raise RuntimeError(
                f"Failed to fetch chunks for {business_doc_id}: {payload}"
            )
        for row in _payload_records(payload):
            chunk_id = str(row["id"])
            content = str(row.get("content") or "").strip()
            if not content or chunk_id in seen or int(row.get("enabled", 1)) != 1:
                continue
            seen.add(chunk_id)
            corpus.append(
                CorpusChunk(
                    chunk_id=chunk_id,
                    business_doc_id=business_doc_id,
                    ragent_doc_id=ragent_doc_id,
                    kb_key=kb_key,
                    kb_id=kb_id,
                    content=content,
                )
            )

    mapped_ragent_ids: set[str] = set()
    for business_doc_id, mapping in sorted(doc_map.items()):
        ragent_doc_id = str(mapping["ragent_doc_id"])
        mapped_ragent_ids.add(ragent_doc_id)
        append_document(
            business_doc_id,
            ragent_doc_id,
            str(mapping["kb_key"]),
            str(mapping["kb_id"]),
        )
    if include_unmapped:
        kb_response = requests.get(
            f"{base_url}/knowledge-base",
            params={"current": 1, "size": 500},
            headers=headers,
            timeout=30,
        )
        kb_response.raise_for_status()
        for kb in _payload_records(kb_response.json()):
            kb_id = str(kb["id"])
            kb_key = str(
                kb.get("collectionName")
                or kb.get("collection_name")
                or f"__unmapped_kb__:{kb_id}"
            )
            docs_response = requests.get(
                f"{base_url}/knowledge-base/{kb_id}/docs",
                params={"current": 1, "size": 500},
                headers=headers,
                timeout=30,
            )
            docs_response.raise_for_status()
            for document in _payload_records(docs_response.json()):
                ragent_doc_id = str(document["id"])
                if ragent_doc_id in mapped_ragent_ids:
                    continue
                append_document(
                    f"__unmapped__:{ragent_doc_id}",
                    ragent_doc_id,
                    kb_key,
                    kb_id,
                )
    if not corpus:
        raise RuntimeError("No enabled chunks were returned by ragent")
    return corpus


def load_or_fetch_corpus(
    *,
    corpus_path: Path,
    base_url: str,
    username: str,
    password: str,
    doc_map_path: Path,
    include_unmapped: bool,
) -> list[CorpusChunk]:
    if corpus_path.exists():
        return [
            CorpusChunk(**row)
            for row in json.loads(corpus_path.read_text(encoding="utf-8"))
        ]
    corpus = fetch_corpus(
        base_url=base_url,
        username=username,
        password=password,
        doc_map_path=doc_map_path,
        include_unmapped=include_unmapped,
    )
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(
        json.dumps([asdict(row) for row in corpus], ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return corpus


def eligible_samples(dataset_path: Path) -> list[EvalSample]:
    return [
        sample
        for sample in load_samples(dataset_path)
        if sample.evaluation_scope == "static-v1"
        and sample.requires_rag
        and sample.expected_doc_ids
    ]


def rank_documents(
    query_vector: np.ndarray,
    document_vectors: np.ndarray,
    corpus: list[CorpusChunk],
    *,
    allowed_kbs: set[str] | None,
    limit: int = 10,
) -> list[str]:
    if allowed_kbs:
        candidate_indices = np.asarray(
            [i for i, row in enumerate(corpus) if row.kb_key in allowed_kbs],
            dtype=np.int64,
        )
    else:
        candidate_indices = np.arange(len(corpus), dtype=np.int64)
    scores = document_vectors[candidate_indices] @ query_vector
    ranked_indices = candidate_indices[np.argsort(-scores)]
    result: list[str] = []
    seen: set[str] = set()
    for index in ranked_indices:
        doc_id = corpus[int(index)].business_doc_id
        if doc_id in seen:
            continue
        result.append(doc_id)
        seen.add(doc_id)
        if len(result) >= limit:
            break
    return result


def retrieval_metrics(
    expected: list[str],
    retrieved: list[str],
) -> dict[str, float]:
    expected_set = set(expected)
    metrics: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        metrics[f"hit@{k}"] = float(bool(expected_set.intersection(retrieved[:k])))
    for k in (5, 10):
        metrics[f"recall@{k}"] = (
            len(expected_set.intersection(retrieved[:k])) / len(expected_set)
        )
    reciprocal_rank = 0.0
    for rank, doc_id in enumerate(retrieved[:10], start=1):
        if doc_id in expected_set:
            reciprocal_rank = 1.0 / rank
            break
    metrics["mrr@10"] = reciprocal_rank
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved[:10], start=1)
        if doc_id in expected_set
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(10, len(expected_set)) + 1)
    )
    metrics["ndcg@10"] = dcg / ideal if ideal else 0.0
    return metrics


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values), q * 100))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@10",
    )
    return {
        name: statistics.fmean(float(row[name]) for row in rows)
        for name in metric_names
    }


def summarize_slices(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {
        name: {"sample_count": len(group), **summarize_rows(group)}
        for name, group in sorted(groups.items())
    }


def content_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def vector_cache_path(
    cache_dir: Path,
    arm: EmbeddingArm,
    *,
    query: bool,
    texts: list[str],
) -> Path:
    prefix_hash = hashlib.sha256(arm.query_prefix.encode("utf-8")).hexdigest()[:10]
    text_hash = content_hash(texts)[:16]
    kind = f"query-{prefix_hash}" if query else "documents"
    safe_model = arm.model.replace("/", "__")
    return cache_dir / f"{safe_model}-{arm.dimension}-{kind}-{text_hash}.npy"


def load_or_embed(
    client: SiliconFlowEmbeddingClient,
    arm: EmbeddingArm,
    texts: list[str],
    *,
    query: bool,
    batch_size: int,
    cache_dir: Path,
    reuse_cache: bool,
) -> tuple[np.ndarray, ApiTelemetry]:
    path = vector_cache_path(cache_dir, arm, query=query, texts=texts)
    if reuse_cache and path.exists():
        vectors = np.load(path)
        if vectors.shape == (len(texts), arm.dimension):
            return vectors, ApiTelemetry(cache_hit=True)
    vectors, telemetry = embed_many(
        client,
        arm,
        texts,
        query=query,
        batch_size=batch_size,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors, telemetry


def telemetry_summary(
    telemetry: ApiTelemetry,
    *,
    item_count: int,
    price_per_million: float | None,
) -> dict[str, Any]:
    elapsed_seconds = sum(telemetry.latencies_ms) / 1000
    estimated_cost = (
        telemetry.prompt_tokens / 1_000_000 * price_per_million
        if price_per_million is not None
        else None
    )
    return {
        "request_count": telemetry.request_count,
        "error_count": telemetry.error_count,
        "error_rate": (
            telemetry.error_count / telemetry.request_count
            if telemetry.request_count
            else 0.0
        ),
        "prompt_tokens": telemetry.prompt_tokens,
        "total_tokens": telemetry.total_tokens,
        "p50_ms": percentile(telemetry.latencies_ms, 0.50),
        "p95_ms": percentile(telemetry.latencies_ms, 0.95),
        "throughput_items_per_second": (
            item_count / elapsed_seconds if elapsed_seconds else None
        ),
        "estimated_cost": estimated_cost,
        "cache_hit": telemetry.cache_hit,
    }


def bootstrap_delta(
    rows_by_arm: dict[str, list[dict[str, Any]]],
    *,
    best_arm: str,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    best_by_id = {row["query_id"]: row for row in rows_by_arm[best_arm]}
    result: dict[str, dict[str, float]] = {}
    rng = np.random.default_rng(seed)
    for arm_id, rows in rows_by_arm.items():
        pairs = [
            (float(row[metric]), float(best_by_id[row["query_id"]][metric]))
            for row in rows
            if row["query_id"] in best_by_id
        ]
        deltas = np.asarray([left - right for left, right in pairs])
        if not len(deltas):
            continue
        samples = rng.choice(deltas, size=(iterations, len(deltas)), replace=True)
        means = samples.mean(axis=1)
        result[arm_id] = {
            "mean_delta": float(deltas.mean()),
            "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)),
        }
    return result


def pricing_from_env() -> dict[str, float]:
    raw = os.environ.get("SILICONFLOW_EMBEDDING_PRICES_JSON", "")
    if not raw:
        return {}
    return {str(key): float(value) for key, value in json.loads(raw).items()}


def storage_plan(dimension: int) -> dict[str, Any]:
    if dimension <= 2000:
        return {
            "storage": f"vector({dimension})",
            "hnsw": True,
            "operator_class": "vector_cosine_ops",
        }
    if dimension <= 4000:
        return {
            "storage": f"halfvec({dimension})",
            "hnsw": True,
            "operator_class": "halfvec_cosine_ops",
            "precision_note": "Validate float16 quantization loss against exact float32.",
        }
    return {
        "storage": f"vector({dimension})",
        "hnsw": False,
        "strategy": "Exact search or lower-dimensional HNSW candidates plus full-vector rerank.",
    }


def collect_pgvector_info(
    *,
    container: str | None,
    database: str,
    user: str,
) -> dict[str, Any]:
    if not container:
        return {"available": False, "reason": "--pg-container not provided"}
    sql = """
SELECT 'extension', extversion FROM pg_extension WHERE extname='vector';
SELECT 'column', format_type(a.atttypid,a.atttypmod)
FROM pg_attribute a
JOIN pg_class c ON c.oid=a.attrelid
WHERE c.relname='t_knowledge_vector' AND a.attname='embedding';
SELECT 'rows', count(*)::text FROM t_knowledge_vector;
SELECT 'index', indexname || E'\\t' || indexdef
FROM pg_indexes WHERE tablename='t_knowledge_vector' ORDER BY indexname;
""".strip()
    try:
        completed = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "psql",
                "-U",
                user,
                "-d",
                database,
                "-At",
                "-F",
                "\t",
                "-c",
                sql,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}
    rows = [line.split("\t", 2) for line in completed.stdout.splitlines()]
    return {"available": True, "rows": rows}


def mrl_probe(
    client: SiliconFlowEmbeddingClient,
    texts: list[str],
) -> dict[str, Any]:
    arm_4096 = EmbeddingArm(
        "mrl-4096",
        "Qwen/Qwen3-Embedding-8B",
        4096,
        True,
        QWEN_QUERY_PREFIX,
        "qwen3",
    )
    arm_1536 = EmbeddingArm(
        "mrl-1536",
        "Qwen/Qwen3-Embedding-8B",
        1536,
        True,
        QWEN_QUERY_PREFIX,
        "qwen3",
    )
    full, full_telemetry = embed_many(
        client, arm_4096, texts, query=True, batch_size=1
    )
    native, native_telemetry = embed_many(
        client, arm_1536, texts, query=True, batch_size=1
    )
    truncated = normalize(full[:, :1536].copy())
    cosines = np.sum(truncated * native, axis=1)
    return {
        "sample_count": len(texts),
        "mean_cosine": float(cosines.mean()),
        "min_cosine": float(cosines.min()),
        "max_abs_difference": float(np.max(np.abs(truncated - native))),
        "interpretation": (
            "A cosine near 1 confirms the API's 1536-dimensional output is "
            "the normalized MRL prefix, not an unrelated projection."
        ),
        "request_count": (
            full_telemetry.request_count + native_telemetry.request_count
        ),
    }


def pro_bge_service_probe(
    client: SiliconFlowEmbeddingClient,
    texts: list[str],
) -> dict[str, Any]:
    standard = EmbeddingArm(
        "bge-m3-standard-probe",
        "BAAI/bge-m3",
        1024,
        False,
        "",
        "bge",
    )
    pro = EmbeddingArm(
        "bge-m3-pro-probe",
        "Pro/BAAI/bge-m3",
        1024,
        False,
        "",
        "bge",
    )
    standard_vectors, standard_telemetry = embed_many(
        client, standard, texts, query=True, batch_size=1
    )
    pro_vectors, pro_telemetry = embed_many(
        client, pro, texts, query=True, batch_size=1
    )
    cosines = np.sum(standard_vectors * pro_vectors, axis=1)
    equivalent = bool(
        float(cosines.min()) >= 0.9999
        and float(np.max(np.abs(standard_vectors - pro_vectors))) <= 0.002
    )
    return {
        "sample_count": len(texts),
        "mean_cosine": float(cosines.mean()),
        "min_cosine": float(cosines.min()),
        "max_abs_difference": float(
            np.max(np.abs(standard_vectors - pro_vectors))
        ),
        "vectors_equivalent": equivalent,
        "quality_handling": (
            "Reuse BAAI/bge-m3 quality results; compare only latency/stability."
            if equivalent
            else "Vectors differ; add Pro/BAAI/bge-m3 as a separate quality arm."
        ),
        "standard": telemetry_summary(
            standard_telemetry,
            item_count=len(texts),
            price_per_million=None,
        ),
        "pro": telemetry_summary(
            pro_telemetry,
            item_count=len(texts),
            price_per_million=None,
        ),
    }


def quality_gate(
    summaries: dict[str, dict[str, Any]],
    rows_by_arm: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    best_hit5 = max(summary["hit@5"] for summary in summaries.values())
    best_mrr = max(summary["mrr@10"] for summary in summaries.values())
    best_by_intent: dict[str, float] = {}
    intent_counts: dict[str, int] = {}
    for rows in rows_by_arm.values():
        for intent, slice_summary in summarize_slices(rows, "intent_l2").items():
            intent_counts[intent] = int(slice_summary["sample_count"])
            best_by_intent[intent] = max(
                best_by_intent.get(intent, 0.0),
                float(slice_summary["hit@5"]),
            )
    decisions: dict[str, Any] = {}
    for arm_id, summary in summaries.items():
        slices = summarize_slices(rows_by_arm[arm_id], "intent_l2")
        slice_regressions = {
            intent: best_by_intent[intent] - float(slice_summary["hit@5"])
            for intent, slice_summary in slices.items()
            if intent_counts.get(intent, 0) >= 5
        }
        reasons = []
        if summary["hit@5"] < 0.97:
            reasons.append("Hit@5 < 97%")
        if summary["recall@5"] < 0.85:
            reasons.append("Recall@5 < 85%")
        if best_hit5 - summary["hit@5"] > 0.01:
            reasons.append("Hit@5 is >1 percentage point below best")
        if best_mrr - summary["mrr@10"] > 0.02:
            reasons.append("MRR@10 is >0.02 below best")
        if any(delta > 0.05 for delta in slice_regressions.values()):
            reasons.append("An intent slice with n>=5 regresses >5 points")
        decisions[arm_id] = {
            "passed": not reasons,
            "reasons": reasons,
            "intent_hit5_regression": slice_regressions,
        }
    return {
        "thresholds": {
            "hit@5": 0.97,
            "recall@5": 0.85,
            "max_hit5_loss": 0.01,
            "max_mrr10_loss": 0.02,
            "max_intent_slice_loss": 0.05,
        },
        "decisions": decisions,
    }


def choose_recommendation(
    summaries: dict[str, dict[str, Any]],
    gate: dict[str, Any],
) -> dict[str, Any]:
    passed = [
        arm_id
        for arm_id, decision in gate["decisions"].items()
        if decision["passed"]
    ]
    if not passed:
        return {
            "selected_arm": None,
            "reason": "No arm passed all configured quality gates.",
        }
    with_cost = [
        arm_id
        for arm_id in passed
        if summaries[arm_id].get("query_cost_per_10k") is not None
    ]
    if with_cost:
        selected = min(
            with_cost,
            key=lambda arm_id: (
                summaries[arm_id]["query_cost_per_10k"],
                summaries[arm_id]["query_embedding"]["p95_ms"] or float("inf"),
                summaries[arm_id]["dimension"],
            ),
        )
        reason = "Lowest estimated cost among arms passing all quality gates."
    else:
        selected = min(
            passed,
            key=lambda arm_id: (
                summaries[arm_id]["query_embedding"]["p95_ms"] or float("inf"),
                summaries[arm_id]["dimension"],
            ),
        )
        reason = (
            "Pricing was unavailable; selected by query P95 then dimension "
            "among arms passing all quality gates."
        )
    return {"selected_arm": selected, "reason": reason, "eligible_arms": passed}


def write_per_query(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "arm_id",
        "scope",
        "query_id",
        "query",
        "intent_l1",
        "intent_l2",
        "difficulty",
        "expected_doc_ids",
        "retrieved_doc_ids",
        "hit@1",
        "hit@3",
        "hit@5",
        "hit@10",
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@10",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["expected_doc_ids"] = json.dumps(
                payload["expected_doc_ids"], ensure_ascii=False
            )
            payload["retrieved_doc_ids"] = json.dumps(
                payload["retrieved_doc_ids"], ensure_ascii=False
            )
            writer.writerow({key: payload.get(key) for key in fieldnames})


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Embedding 模型选型报告",
        "",
        f"- 样本数：{result['sample_count']}",
        f"- Chunk 数：{result['chunk_count']}",
        f"- 主判定作用域：{result['primary_scope']}",
        f"- 推荐配置：{result['recommendation'].get('selected_arm') or '无'}",
        f"- 判定说明：{result['recommendation']['reason']}",
        "",
        "## 质量与工程指标",
        "",
        "| Arm | 维度 | Hit@5 | Recall@5 | MRR@10 | Query P95(ms) | 每万次查询成本 | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for arm_id, summary in result["summaries"].items():
        decision = result["quality_gate"]["decisions"][arm_id]
        cost = summary.get("query_cost_per_10k")
        lines.append(
            f"| {arm_id} | {summary['dimension']} | "
            f"{summary['hit@5']:.3f} | {summary['recall@5']:.3f} | "
            f"{summary['mrr@10']:.3f} | "
            f"{summary['query_embedding']['p95_ms'] or 0:.1f} | "
            f"{cost:.4f} | {'PASS' if decision['passed'] else 'FAIL'} |"
            if cost is not None
            else
            f"| {arm_id} | {summary['dimension']} | "
            f"{summary['hit@5']:.3f} | {summary['recall@5']:.3f} | "
            f"{summary['mrr@10']:.3f} | "
            f"{summary['query_embedding']['p95_ms'] or 0:.1f} | n/a | "
            f"{'PASS' if decision['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "- 质量检索为 chunk 级精确余弦排序，再按文档 ID 去重；未启用关键词、RRF 或 rerank。",
            "- `global` 为全库检索，`gold_collection` 仅在 gold 文档所属知识库中检索。",
            "- BGE 请求不发送 `dimensions`；Qwen 请求显式发送目标维度。",
            "- 2048 维以上的索引方案与精度风险见 `matrix.json` 的 `storage_plan`。",
        ]
    )
    if result.get("mrl_probe"):
        probe = result["mrl_probe"]
        lines.extend(
            [
                "",
                "## MRL 探针",
                "",
                f"- 4096 前 1536 维归一化 vs API 原生 1536 平均余弦：{probe['mean_cosine']:.8f}",
                f"- 最低余弦：{probe['min_cosine']:.8f}",
                f"- 最大绝对差：{probe['max_abs_difference']:.8f}",
            ]
        )
    if result.get("pro_bge_service_probe"):
        probe = result["pro_bge_service_probe"]
        lines.extend(
            [
                "",
                "## BGE-M3 服务探针",
                "",
                f"- 普通版 vs Pro 最低余弦：{probe['min_cosine']:.8f}",
                f"- 向量等价：{'是' if probe['vectors_equivalent'] else '否'}",
                f"- 处理方式：{probe['quality_handling']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_project_env()
    api_key = os.environ.get("SILICONFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Missing SILICONFLOW_API_KEY")
    output_dir = Path(args.output or (
        DEFAULT_REPORTS_DIR
        / f"embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "vectors"
    dataset_path = Path(args.dataset)
    doc_map_path = Path(args.doc_map)
    corpus_path = Path(args.corpus) if args.corpus else output_dir / "corpus.json"
    base_url = os.environ.get("RAGENT_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    username = os.environ.get("RAGENT_USERNAME", "admin")
    password = os.environ.get("RAGENT_PASSWORD", "admin")
    corpus = load_or_fetch_corpus(
        corpus_path=corpus_path,
        base_url=base_url,
        username=username,
        password=password,
        doc_map_path=doc_map_path,
        include_unmapped=args.include_unmapped,
    )
    samples = eligible_samples(dataset_path)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise RuntimeError("No eligible static retrieval samples")
    doc_map = json.loads(doc_map_path.read_text(encoding="utf-8"))
    arms = load_arms(args.preset, Path(args.matrix) if args.matrix else None)
    random.Random(args.seed).shuffle(arms)
    client = SiliconFlowEmbeddingClient(
        api_key=api_key,
        base_url=os.environ.get(
            "SILICONFLOW_BASE_URL", DEFAULT_SILICONFLOW_URL
        ),
        timeout=args.timeout,
        retries=args.retries,
    )
    prices = pricing_from_env()
    document_texts = [row.content for row in corpus]
    query_texts = [sample.query for sample in samples]
    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    primary_rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for index, arm in enumerate(arms, start=1):
        print(
            f"[{index}/{len(arms)}] {arm.arm_id}: "
            f"{arm.model} / {arm.dimension}"
        )
        try:
            document_vectors, document_telemetry = load_or_embed(
                client,
                arm,
                document_texts,
                query=False,
                batch_size=args.batch_size,
                cache_dir=cache_dir,
                reuse_cache=args.reuse_cache,
            )
            query_vectors, query_telemetry = load_or_embed(
                client,
                arm,
                query_texts,
                query=True,
                batch_size=1,
                cache_dir=cache_dir,
                reuse_cache=args.reuse_cache,
            )
            for _ in range(1, args.latency_rounds):
                _, extra = embed_many(
                    client,
                    arm,
                    query_texts,
                    query=True,
                    batch_size=1,
                )
                query_telemetry.merge(extra)
            scope_rows: dict[str, list[dict[str, Any]]] = {
                "global": [],
                "gold_collection": [],
            }
            for sample, query_vector in zip(samples, query_vectors):
                gold_kbs = {
                    str(doc_map[doc_id]["kb_key"])
                    for doc_id in sample.expected_doc_ids
                    if doc_id in doc_map
                }
                for scope, allowed_kbs in (
                    ("global", None),
                    ("gold_collection", gold_kbs),
                ):
                    retrieved = rank_documents(
                        query_vector,
                        document_vectors,
                        corpus,
                        allowed_kbs=allowed_kbs,
                    )
                    row = {
                        "arm_id": arm.arm_id,
                        "scope": scope,
                        "query_id": sample.query_id,
                        "query": sample.query,
                        "intent_l1": sample.intent_l1,
                        "intent_l2": sample.intent_l2,
                        "difficulty": sample.difficulty,
                        "expected_doc_ids": sample.expected_doc_ids,
                        "retrieved_doc_ids": retrieved,
                        **retrieval_metrics(sample.expected_doc_ids, retrieved),
                    }
                    scope_rows[scope].append(row)
                    all_rows.append(row)
                    if row["hit@5"] == 0 or row["recall@5"] < 1:
                        failures.append(row)
            primary_rows = scope_rows[args.primary_scope]
            primary_rows_by_arm[arm.arm_id] = primary_rows
            price = prices.get(arm.model)
            query_summary = telemetry_summary(
                query_telemetry,
                item_count=len(samples) * args.latency_rounds,
                price_per_million=price,
            )
            query_cost_per_10k = None
            if price is not None and query_telemetry.prompt_tokens:
                average_tokens = query_telemetry.prompt_tokens / (
                    len(samples) * args.latency_rounds
                )
                query_cost_per_10k = average_tokens * 10_000 / 1_000_000 * price
            pgvector_validation = None
            if args.pgvector_validation_database:
                if not args.pg_container:
                    raise RuntimeError(
                        "--pgvector-validation-database requires --pg-container"
                    )
                from eval.rag.pgvector_validation import validate_pgvector

                try:
                    pgvector_validation = validate_pgvector(
                        container=args.pg_container,
                        database=args.pgvector_validation_database,
                        user=args.pg_user,
                        document_vectors=document_vectors,
                        query_vectors=query_vectors,
                        query_limit=args.pgvector_query_limit,
                        candidate_k=args.pgvector_candidate_k,
                    )
                except Exception as exc:  # noqa: BLE001
                    pgvector_validation = {"error": str(exc)}
            summaries[arm.arm_id] = {
                "model": arm.model,
                "dimension": arm.dimension,
                "send_dimensions": arm.send_dimensions,
                "query_prefix": arm.query_prefix,
                "note": arm.note,
                "max_input_chars": arm.max_input_chars,
                **summarize_rows(primary_rows),
                "by_scope": {
                    scope: summarize_rows(rows)
                    for scope, rows in scope_rows.items()
                },
                "by_intent": summarize_slices(primary_rows, "intent_l2"),
                "by_difficulty": summarize_slices(primary_rows, "difficulty"),
                "document_embedding": telemetry_summary(
                    document_telemetry,
                    item_count=len(corpus),
                    price_per_million=price,
                ),
                "query_embedding": query_summary,
                "query_cost_per_10k": query_cost_per_10k,
                "storage_plan": storage_plan(arm.dimension),
                "pgvector_validation": pgvector_validation,
            }
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {"arm_id": arm.arm_id, "scope": "arm", "error": str(exc)}
            )
            print(f"  FAILED: {exc}")
    if not summaries:
        raise RuntimeError("All benchmark arms failed")
    best_arm = max(
        summaries,
        key=lambda arm_id: (
            summaries[arm_id]["hit@5"],
            summaries[arm_id]["mrr@10"],
        ),
    )
    bootstrap = bootstrap_delta(
        primary_rows_by_arm,
        best_arm=best_arm,
        metric="mrr@10",
        iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    gate = quality_gate(summaries, primary_rows_by_arm)
    result: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": str(dataset_path.resolve()),
        "doc_map": str(doc_map_path.resolve()),
        "corpus": str(corpus_path.resolve()),
        "sample_count": len(samples),
        "chunk_count": len(corpus),
        "document_count": len({row.business_doc_id for row in corpus}),
        "mapped_chunk_count": sum(
            not row.business_doc_id.startswith("__unmapped__:") for row in corpus
        ),
        "unmapped_chunk_count": sum(
            row.business_doc_id.startswith("__unmapped__:") for row in corpus
        ),
        "primary_scope": args.primary_scope,
        "latency_rounds": args.latency_rounds,
        "execution_order": [arm.arm_id for arm in arms],
        "arms": [asdict(arm) for arm in arms],
        "summaries": summaries,
        "bootstrap_mrr10_vs_best": {
            "best_arm": best_arm,
            "iterations": args.bootstrap_iterations,
            "deltas": bootstrap,
        },
        "quality_gate": gate,
        "recommendation": choose_recommendation(summaries, gate),
        "database_index_info": collect_pgvector_info(
            container=args.pg_container,
            database=args.pg_database,
            user=args.pg_user,
        ),
    }
    if args.mrl_probe:
        result["mrl_probe"] = mrl_probe(
            client, query_texts[: min(5, len(query_texts))]
        )
    if args.pro_bge_probe:
        result["pro_bge_service_probe"] = pro_bge_service_probe(
            client, query_texts[: min(5, len(query_texts))]
        )
    (output_dir / "matrix.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_per_query(output_dir / "per_query.csv", all_rows)
    with (output_dir / "failures.jsonl").open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(output_dir / "report.md", result)
    print(f"output={output_dir.resolve()}")
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preset",
        choices=["semantic", "dimensions", "all"],
        default="semantic",
    )
    parser.add_argument("--matrix", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--doc-map", type=Path, default=DEFAULT_DOC_MAP)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latency-rounds", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--primary-scope",
        choices=["global", "gold_collection"],
        default="gold_collection",
    )
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument(
        "--include-unmapped",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="将当前数据库中未出现在 doc-map 的文档作为全库检索干扰项",
    )
    parser.add_argument("--mrl-probe", action="store_true")
    parser.add_argument(
        "--pro-bge-probe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="比较普通与 Pro BGE-M3 的向量一致性、延迟和稳定性",
    )
    parser.add_argument(
        "--pg-container",
        default=os.environ.get("RAGENT_POSTGRES_CONTAINER"),
    )
    parser.add_argument("--pg-database", default="ragent")
    parser.add_argument("--pg-user", default="postgres")
    parser.add_argument(
        "--pgvector-validation-database",
        default=None,
        help="专用实验数据库；拒绝 ragent/postgres 等共享数据库",
    )
    parser.add_argument("--pgvector-query-limit", type=int, default=20)
    parser.add_argument("--pgvector-candidate-k", type=int, default=50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
