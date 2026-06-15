"""Step 1: 在 ragent 中创建 4 个知识库。

行为：
    1. POST /auth/login 拿到 sa-token
    2. 顺序创建 4 个 KB（商品 / 手册 / 政策 / FAQ）
    3. 落地 eval/rag/init/kb_ids.json 供 Step 2 复用

幂等性：本脚本不做幂等，重复执行会重复创建同名 KB。需要清理时手动调 DELETE。

环境变量：
    RAGENT_BASE_URL   默认 http://localhost:9090/api/ragent
    RAGENT_USERNAME   必填
    RAGENT_PASSWORD   必填
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://localhost:9090/api/ragent"

KB_SPECS: list[dict[str, str]] = [
    {
        "key": "product",
        "name": "比特严选-商品库",
        "collection_name": "kb-product",
    },
    {
        "key": "manual",
        "name": "比特严选-使用手册库",
        "collection_name": "kb-manual",
    },
    {
        "key": "policy",
        "name": "比特严选-政策库",
        "collection_name": "kb-policy",
    },
    {
        "key": "faq",
        "name": "比特严选-FAQ库",
        "collection_name": "kb-faq",
    },
]

EMBEDDING_MODEL = "qwen-emb-8b"

OUTPUT_PATH = Path(__file__).resolve().parent / "kb_ids.json"


def http_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """发起 JSON 请求，返回反序列化后的响应体。"""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = token
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} -> HTTP {exc.code}: {detail}") from exc
    if not payload:
        return {}
    return json.loads(payload)


def login(base_url: str, username: str, password: str) -> str:
    """登录并返回 token。"""
    resp = http_json(
        f"{base_url}/auth/login",
        method="POST",
        body={"username": username, "password": password},
    )
    if not resp.get("success", True):
        raise RuntimeError(f"登录失败：{resp}")
    data = resp.get("data") or {}
    token = data.get("token")
    if not token:
        raise RuntimeError(f"登录响应里没拿到 token：{resp}")
    return token


def create_kb(
    base_url: str,
    token: str,
    name: str,
    collection_name: str,
    embedding_model: str,
) -> str:
    """创建一个 KB，返回 kb_id。"""
    resp = http_json(
        f"{base_url}/knowledge-base",
        method="POST",
        body={
            "name": name,
            "embeddingModel": embedding_model,
            "collectionName": collection_name,
        },
        token=token,
    )
    if not resp.get("success", True):
        raise RuntimeError(f"创建 KB '{name}' 失败：{resp}")
    kb_id = resp.get("data")
    if not kb_id:
        raise RuntimeError(f"创建 KB '{name}' 响应里没拿到 id：{resp}")
    return kb_id


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 1: 创建隔离知识库")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--dimension", type=int, default=1536)
    parser.add_argument(
        "--collection-prefix",
        default="",
        help="为 collection/bucket 名增加实验前缀，避免隔离环境之间冲突",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="写入 kb_ids.json 和 experiment.json 的隔离目录",
    )
    args = parser.parse_args()
    state_dir = Path(args.state_dir) if args.state_dir else OUTPUT_PATH.parent
    state_dir.mkdir(parents=True, exist_ok=True)
    output_path = state_dir / "kb_ids.json"
    base_url = os.environ.get("RAGENT_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    username = os.environ.get("RAGENT_USERNAME")
    password = os.environ.get("RAGENT_PASSWORD")
    if not username or not password:
        print("缺少环境变量 RAGENT_USERNAME / RAGENT_PASSWORD", file=sys.stderr)
        return 2

    print(f"[1/2] 登录 {base_url} ...")
    token = login(base_url, username, password)
    print("      OK")

    print(f"[2/2] 创建 {len(KB_SPECS)} 个 KB ...")
    results: dict[str, dict[str, str]] = {}
    for spec in KB_SPECS:
        collection_name = (
            f"{args.collection_prefix}-{spec['collection_name']}"
            if args.collection_prefix
            else spec["collection_name"]
        )
        kb_id = create_kb(
            base_url,
            token,
            spec["name"],
            collection_name,
            args.embedding_model,
        )
        print(f"      {spec['key']:>8s}  '{spec['name']}'  ->  {kb_id}")
        results[spec["key"]] = {
            "kb_id": kb_id,
            "name": spec["name"],
            "collection_name": collection_name,
            "embedding_model": args.embedding_model,
            "embedding_dimension": args.dimension,
        }

    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (state_dir / "experiment.json").write_text(
        json.dumps(
            {
                "embedding_model": args.embedding_model,
                "dimension": args.dimension,
                "ragent_base_url": base_url,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"\n写入 {output_path.relative_to(Path.cwd()) if output_path.is_relative_to(Path.cwd()) else output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
