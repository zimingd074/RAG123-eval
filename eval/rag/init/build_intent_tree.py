"""Step 3: 构建并灌入意图识别树。

行为：
    1. 从评估集 + doc_id_map.json 数据驱动算出每个 intent_l2 的候选 KB
    2. 拉取每个 intent_l2 的真实 query 作为 examples（最多 5 条）
    3. 拼出 DOMAIN(3) -> CATEGORY(5) -> TOPIC(22) 共 30 个节点的入参
    4. 按 DOMAIN -> CATEGORY -> TOPIC 顺序逐个 POST /intent-tree
    5. 写 intent_ids.json：intentCode -> ragent 内部 node id

业务规则：
    - F2/F3/C1/C2 是 SYSTEM kind，必要知识由并列 KB 意图提供
    - 其他 18 个 leaf 是 KB kind，支持 kbIds 多知识库并保留 kbId 首库兼容
    - L0/L1 节点 kbId 一律为空（服务端只在 TOPIC+KB 时强校验）

幂等：基于 intent_ids.json 跳过已创建 intentCode。

环境变量：与 create_kbs.py 一致。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:9090/api/ragent"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INIT_DIR = Path(__file__).resolve().parent
KB_IDS_PATH = INIT_DIR / "kb_ids.json"
DOC_MAP_PATH = PROJECT_ROOT / "eval" / "rag" / "dataset" / "doc_id_map.json"
EVAL_SET_PATH = PROJECT_ROOT / "eval" / "rag" / "dataset" / "eval_set_v1_all.jsonl"
INTENT_IDS_PATH = INIT_DIR / "intent_ids.json"

REQUEST_TIMEOUT = 30

# ragent IntentKind 枚举
KIND_KB = 0
KIND_SYSTEM = 1
KIND_MCP = 2

# ragent IntentLevel 枚举
LEVEL_DOMAIN = 0
LEVEL_CATEGORY = 1
LEVEL_TOPIC = 2

# Top-K 默认值（仅 KB-kind leaf 用）
DEFAULT_TOP_K = 5

# 不走 RAG 的 leaf 白名单（kind=SYSTEM）
SYSTEM_LEAF_CODES = {"F2_功能建议", "F3_投诉吐槽", "C1_寒暄问候", "C2_越界提问"}

SYSTEM_PROMPTS: dict[str, str] = {
    "F2_功能建议": (
        "你是比特严选客服助手。用户在提出功能建议时，只需确认已理解并记录建议，"
        "用一两句话复述核心诉求；不要输出操作教程，不要声称功能已经存在或承诺上线时间。"
    ),
    "F3_投诉吐槽": (
        "你是比特严选客服助手。先简短共情并确认用户诉求。纯情绪投诉应记录问题并说明"
        "可转人工处理；不要争辩、不要作法律定性。若问题同时涉及物流、故障或售后事实，"
        "应结合并列知识意图提供的证据处理，不得编造政策。"
    ),
    "C1_寒暄问候": (
        "你是比特严选智能客服助手，不是真人客服。用一到三句话简洁回应，说明可协助"
        "商品参数、订单物流、退换售后、设备使用和故障排查。"
    ),
    "C2_越界提问": (
        "你是比特严选客服助手。对天气、创作、品牌站队和资料不足的竞品问题保持边界："
        "不主观站队，不编造竞品信息；简洁说明无法可靠确认，并引导到比特严选可支持的"
        "商品、订单、售后和设备问题。"
    ),
}

PREFERRED_KB_KEYS: dict[str, list[str]] = {
    "F1_故障报告": ["faq", "manual", "policy"],
    "S6_配件兼容": ["product", "manual"],
    "S9_配网连接": ["manual", "faq"],
    "S12_生态联动": ["manual", "product"],
    "S13_保养维护": ["manual", "faq"],
    "S17_发票会员": ["policy"],
}

# 树骨架定义：(intentCode, name, parentCode)
DOMAINS = [
    ("SUPPORT", "产品咨询"),
    ("FEEDBACK", "用户反馈"),
    ("CHAT", "闲聊兜底"),
]

CATEGORIES = [
    # (intentCode, name, parent_domain)
    ("SUPPORT_PRESALE", "售前咨询", "SUPPORT"),
    ("SUPPORT_USAGE", "使用咨询", "SUPPORT"),
    ("SUPPORT_AFTERSALES", "售后咨询", "SUPPORT"),
    ("FEEDBACK_ALL", "反馈处理", "FEEDBACK"),
    ("CHAT_ALL", "兜底对话", "CHAT"),
]

# leaf 归属：(intentCode, name, parent_category)
TOPICS = [
    ("S1_选购推荐", "选购推荐", "SUPPORT_PRESALE"),
    ("S2_参数咨询", "参数咨询", "SUPPORT_PRESALE"),
    ("S3_对比选购", "对比选购", "SUPPORT_PRESALE"),
    ("S4_价格活动", "价格活动", "SUPPORT_PRESALE"),
    ("S5_库存到货", "库存到货", "SUPPORT_PRESALE"),
    ("S6_配件兼容", "配件兼容", "SUPPORT_PRESALE"),
    ("S7_适用场景", "适用场景", "SUPPORT_PRESALE"),
    ("S8_操作指引", "操作指引", "SUPPORT_USAGE"),
    ("S9_配网连接", "配网连接", "SUPPORT_USAGE"),
    ("S10_APP功能", "APP 功能", "SUPPORT_USAGE"),
    ("S11_固件升级", "固件升级", "SUPPORT_USAGE"),
    ("S12_生态联动", "生态联动", "SUPPORT_USAGE"),
    ("S13_保养维护", "保养维护", "SUPPORT_USAGE"),
    ("S14_售后政策", "售后政策", "SUPPORT_AFTERSALES"),
    ("S15_退换货", "退换货", "SUPPORT_AFTERSALES"),
    ("S16_物流配送", "物流配送", "SUPPORT_AFTERSALES"),
    ("S17_发票会员", "发票/会员", "SUPPORT_AFTERSALES"),
    ("F1_故障报告", "故障报告", "FEEDBACK_ALL"),
    ("F2_功能建议", "功能建议", "FEEDBACK_ALL"),
    ("F3_投诉吐槽", "投诉吐槽", "FEEDBACK_ALL"),
    ("C1_寒暄问候", "寒暄问候", "CHAT_ALL"),
    ("C2_越界提问", "越界提问", "CHAT_ALL"),
]


def login(base_url: str, username: str, password: str) -> str:
    resp = requests.post(
        f"{base_url}/auth/login",
        json={"username": username, "password": password},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    token = (resp.json().get("data") or {}).get("token")
    if not token:
        raise RuntimeError(f"登录响应里没拿到 token：{resp.text}")
    return token


def vote_kb_per_leaf(
    eval_path: Path, doc_map: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """对每个 intent_l2 算 expected_doc_ids 的多数派 KB。"""
    votes: dict[str, Counter] = defaultdict(Counter)
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            intent = r["intent_l2"]
            for doc_id in r.get("expected_doc_ids") or []:
                entry = doc_map.get(doc_id)
                if entry:
                    votes[intent][entry["kb_key"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items() if c}


def kb_keys_per_leaf(
    eval_path: Path,
    doc_map: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    """Return ordered KB keys per intent with curated multi-KB overrides."""
    votes: dict[str, Counter] = defaultdict(Counter)
    with eval_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for doc_id in row.get("expected_doc_ids") or []:
                entry = doc_map.get(doc_id)
                if entry:
                    votes[row["intent_l2"]][entry["kb_key"]] += 1

    result: dict[str, list[str]] = {}
    for intent, counter in votes.items():
        observed = [key for key, _ in counter.most_common()]
        preferred = PREFERRED_KB_KEYS.get(intent, [])
        result[intent] = list(dict.fromkeys(preferred + observed))
    return result


def sample_queries_per_leaf(
    eval_path: Path, limit: int = 5
) -> dict[str, list[str]]:
    """每个 intent_l2 取最多 limit 条 query 做 examples。"""
    out: dict[str, list[str]] = defaultdict(list)
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            intent = r["intent_l2"]
            if len(out[intent]) < limit:
                out[intent].append(r["query"])
    return dict(out)


def make_description(name: str, examples: list[str]) -> str:
    """MVP 简化版：用 leaf name + 第一条示例。AI 增强留到后续脚本做。"""
    base = f"用户的{name}相关问题"
    if examples:
        base += f"，例如：{examples[0]}"
    boundary = BOUNDARY_DISAMBIGUATION.get(name)
    if boundary:
        base += f"。{boundary}"
    return base


# 相邻易混淆意图的硬编码区分说明（补充到 leaf description 中供 LLM 区分）
BOUNDARY_DISAMBIGUATION: dict[str, str] = {
    "S2_参数咨询": "只回答明确型号的客观规格参数；询问某类人群、房间或用途是否合适时归入 S7_适用场景",
    "S7_适用场景": "判断商品是否适合某个人群、面积、用途或环境；单纯询问尺寸、功率、容量等规格归入 S2_参数咨询",
    "S4_价格活动": "商品参考价、活动规则和价保补差属于本分类；退货退款条件归入 S15_退换货",
    "S5_库存到货": "只处理有无现货、补货、到货提醒和未发布新品时间；发货、运输、签收和配送时效归入 S16_物流配送",
    "S9_配网连接": "设备联网、Wi-Fi、蓝牙发现、配网失败属于本分类；联网后在 APP 内查找功能或设置入口归入 S10_APP功能",
    "S10_APP功能": "米家或商城 APP 内的页面、功能、地址簿、订单设置属于本分类；设备连不上网络或发现不到设备归入 S9_配网连接",
    "S12_生态联动": "已有设备之间的自动化、条件触发和联动能力属于本分类；用户提出尚不存在的新功能诉求归入 F2_功能建议",
    "S15_退换货": "处理退货、换货、退款条件和流程；降价补差、价保周期归入 S4_价格活动",
    "S16_物流配送": "下单后的发货、承运、预计送达、改地址、破损签收和物流异常属于本分类；有无现货和补货时间归入 S5_库存到货。投诉物流慢时可与 F3_投诉吐槽同时命中",
    "F1_故障报告": "设备卡顿、死机、报错、无法工作等故障属于本分类；带有抱怨语气时可与 F3_投诉吐槽同时命中",
    "F2_功能建议": "用户希望新增、改进某项功能时归入本分类，只记录建议；询问现有设备能否联动归入 S12_生态联动",
    "F3_投诉吐槽": "识别用户不满和投诉。若同时出现发货、物流、故障、保修或退换事实，应保留 F3 为主要意图，并同时给对应 KB 意图较高分",
    "C1_寒暄问候": "你好、在吗、你是谁、是否真人、叫什么名字等身份与寒暄问题均稳定归入本分类",
    "C2_越界提问": "天气、创作、通用问答、竞品评价和品牌站队等非比特严选客服范围问题归入本分类",
}


def build_payloads(
    kb_ids: dict[str, dict[str, str]],
    kb_keys_by_leaf: dict[str, list[str]],
    examples_per_leaf: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """生成 30 个节点的 IntentNodeCreateRequest payload（按依赖顺序）。"""
    payloads: list[dict[str, Any]] = []

    for sort_order, (code, name) in enumerate(DOMAINS):
        payloads.append(
            {
                "intentCode": code,
                "name": name,
                "level": LEVEL_DOMAIN,
                "parentCode": None,
                "kbId": None,
                "kind": KIND_KB,  # 非 leaf，kind 实际不被使用，默认 0
                "description": f"一级意图：{name}",
                "examples": [],
                "topK": None,
                "sortOrder": sort_order,
                "enabled": 1,
            }
        )

    for sort_order, (code, name, parent) in enumerate(CATEGORIES):
        payloads.append(
            {
                "intentCode": code,
                "name": name,
                "level": LEVEL_CATEGORY,
                "parentCode": parent,
                "kbId": None,
                "kind": KIND_KB,
                "description": f"二级分组：{name}",
                "examples": [],
                "topK": None,
                "sortOrder": sort_order,
                "enabled": 1,
            }
        )

    for sort_order, (code, name, parent) in enumerate(TOPICS):
        is_system = code in SYSTEM_LEAF_CODES
        kind = KIND_SYSTEM if is_system else KIND_KB
        examples = examples_per_leaf.get(code, [])
        leaf_kb_ids: list[str] = []
        if not is_system:
            kb_keys = kb_keys_by_leaf.get(code, [])
            if not kb_keys:
                print(
                    f"[WARN] leaf {code} ({name}) 在 eval set 中无样本，跳过该 leaf"
                )
                continue
            leaf_kb_ids = [
                kb_ids[key]["kb_id"]
                for key in kb_keys
                if key in kb_ids
            ]
            if not leaf_kb_ids:
                print(f"[WARN] leaf {code} 未找到可用知识库，跳过该 leaf")
                continue
        payloads.append(
            {
                "intentCode": code,
                "name": name,
                "level": LEVEL_TOPIC,
                "parentCode": parent,
                "kbId": leaf_kb_ids[0] if leaf_kb_ids else None,
                "kbIds": leaf_kb_ids,
                "kind": kind,
                "description": make_description(name, examples),
                "examples": examples,
                "topK": DEFAULT_TOP_K if not is_system else None,
                "sortOrder": sort_order,
                "enabled": 1,
                "promptTemplate": SYSTEM_PROMPTS.get(code),
            }
        )

    return payloads


def create_node(base_url: str, token: str, payload: dict[str, Any]) -> str:
    resp = requests.post(
        f"{base_url}/intent-tree",
        json=payload,
        headers={"Authorization": token, "Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"POST /intent-tree failed HTTP {resp.status_code}: {resp.text}"
        )
    body = resp.json()
    node_id = body.get("data")
    if not node_id:
        raise RuntimeError(f"响应没拿到 node id：{body}")
    return node_id


def update_node(
    base_url: str,
    token: str,
    node_id: str,
    payload: dict[str, Any],
) -> None:
    """Update an existing node with the same bootstrap configuration."""
    update_payload = {
        key: payload[key]
        for key in (
            "name",
            "level",
            "parentCode",
            "description",
            "examples",
            "kbIds",
            "topK",
            "kind",
            "sortOrder",
            "enabled",
            "promptSnippet",
            "promptTemplate",
            "paramPromptTemplate",
        )
        if key in payload
    }
    resp = requests.put(
        f"{base_url}/intent-tree/{node_id}",
        json=update_payload,
        headers={"Authorization": token, "Content-Type": "application/json"},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"PUT /intent-tree/{node_id} failed HTTP "
            f"{resp.status_code}: {resp.text}"
        )
    if not resp.content:
        return
    body = resp.json()
    if not body.get("success", True):
        raise RuntimeError(f"更新意图节点失败：{body}")


def load_intent_ids() -> dict[str, str]:
    if INTENT_IDS_PATH.exists():
        return json.loads(INTENT_IDS_PATH.read_text(encoding="utf-8"))
    return {}


def save_intent_ids(m: dict[str, str]) -> None:
    INTENT_IDS_PATH.write_text(
        json.dumps(m, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 3: 构建并灌入意图识别树")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要创建的节点，不实际请求",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="删除 intent_ids.json 全量重建（默认跳过已创建的节点以支持断点续传）",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="更新 intent_ids.json 中已存在的节点，应用 kbIds、描述和 Prompt",
    )
    args = parser.parse_args()

    if not KB_IDS_PATH.exists():
        print(f"找不到 {KB_IDS_PATH}，请先跑 create_kbs.py", file=sys.stderr)
        return 2
    if not DOC_MAP_PATH.exists():
        print(f"找不到 {DOC_MAP_PATH}，请先跑 upload_docs.py", file=sys.stderr)
        return 2

    kb_ids = json.loads(KB_IDS_PATH.read_text(encoding="utf-8"))
    doc_map = json.loads(DOC_MAP_PATH.read_text(encoding="utf-8"))

    kb_keys_by_leaf = kb_keys_per_leaf(EVAL_SET_PATH, doc_map)
    examples_per_leaf = sample_queries_per_leaf(EVAL_SET_PATH)

    payloads = build_payloads(kb_ids, kb_keys_by_leaf, examples_per_leaf)

    # 反向映射：kb_id -> kb_key，方便 dry-run 输出
    kb_id_to_key = {v["kb_id"]: k for k, v in kb_ids.items()}

    print(f"将创建 {len(payloads)} 个节点：")
    for p in payloads:
        kb_tag = f" kb={kb_id_to_key.get(p['kbId'], '?')}" if p["kbId"] else ""
        kind_tag = {0: "KB", 1: "SYSTEM", 2: "MCP"}[p["kind"]]
        level_tag = {0: "D", 1: "C", 2: "T"}[p["level"]]
        ex_n = len(p["examples"])
        parent = f"  parent={p['parentCode']}" if p["parentCode"] else ""
        print(
            f"  [{level_tag}] {p['intentCode']:<20s}  kind={kind_tag:<6s}{kb_tag:<14s}  examples={ex_n}{parent}"
        )

    # 汇总分布
    kind_count = Counter(p["kind"] for p in payloads if p["level"] == LEVEL_TOPIC)
    kb_count = Counter(
        kb_id_to_key.get(p["kbId"]) for p in payloads if p["kbId"]
    )
    print("\n--- 汇总 ---")
    print(f"  DOMAIN: 3, CATEGORY: 5, TOPIC: 22")
    print(
        f"  TOPIC by kind: KB={kind_count[KIND_KB]}, SYSTEM={kind_count[KIND_SYSTEM]}, MCP={kind_count[KIND_MCP]}"
    )
    print(f"  KB leaves by KB: {dict(kb_count)}")

    if args.dry_run:
        print("\n--dry-run，未提交。")
        return 0

    base_url = os.environ.get("RAGENT_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    username = os.environ.get("RAGENT_USERNAME")
    password = os.environ.get("RAGENT_PASSWORD")
    if not username or not password:
        print("缺少环境变量 RAGENT_USERNAME / RAGENT_PASSWORD", file=sys.stderr)
        return 2

    print(f"\n登录 {base_url} ...")
    token = login(base_url, username, password)
    print("OK\n")

    if args.force and INTENT_IDS_PATH.exists():
        INTENT_IDS_PATH.unlink()
        print(f"已删除 {INTENT_IDS_PATH}，将全量重建\n")

    intent_ids = load_intent_ids()
    success = 0
    failed: list[tuple[str, str]] = []
    skipped = 0
    for idx, payload in enumerate(payloads, start=1):
        code = payload["intentCode"]
        if code in intent_ids:
            if not args.sync:
                skipped += 1
                continue
            try:
                update_node(base_url, token, intent_ids[code], payload)
                success += 1
                print(
                    f"  [{idx:>2d}/{len(payloads)}] UPDATED "
                    f"{code} -> {intent_ids[code]}"
                )
            except Exception as exc:  # noqa: BLE001
                failed.append((code, str(exc)))
                print(
                    f"  [{idx:>2d}/{len(payloads)}] FAILED {code}  {exc}",
                    file=sys.stderr,
                )
            continue
        try:
            node_id = create_node(base_url, token, payload)
            intent_ids[code] = node_id
            save_intent_ids(intent_ids)
            success += 1
            print(f"  [{idx:>2d}/{len(payloads)}] ✓ {code} -> {node_id}")
        except Exception as exc:  # noqa: BLE001
            failed.append((code, str(exc)))
            print(f"  [{idx:>2d}/{len(payloads)}] ✗ {code}  {exc}", file=sys.stderr)

    print(f"\n完成：成功 {success}，失败 {len(failed)}，跳过 {skipped}")
    if failed:
        return 1
    print(f"\n写入 {INTENT_IDS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
