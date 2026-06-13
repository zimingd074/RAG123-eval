"""Tests for profile-aware intent tree bootstrap configuration."""
from __future__ import annotations

import unittest

from eval.rag.init.build_intent_tree import (
    EVAL_SET_PATH,
    PREFERRED_KB_KEYS,
    SYSTEM_PROMPTS,
    build_payloads,
    kb_keys_per_leaf,
    sample_queries_per_leaf,
)


class IntentTreeConfigTest(unittest.TestCase):
    """Verify multi-KB and SYSTEM prompt bootstrap payloads."""

    @staticmethod
    def fake_kb_ids() -> dict[str, dict[str, str]]:
        return {
            key: {"kb_id": f"{key}-id"}
            for key in ("product", "manual", "policy", "faq")
        }

    @staticmethod
    def fake_doc_map() -> dict[str, dict[str, str]]:
        """Map all observed docs to product; curated overrides add target KBs."""
        import json

        rows = [
            json.loads(line)
            for line in EVAL_SET_PATH.read_text(encoding="utf-8").splitlines()
            if line
        ]
        return {
            doc_id: {"kb_key": "product"}
            for row in rows
            for doc_id in row.get("expected_doc_ids") or []
        }

    def test_curated_intents_emit_multiple_kb_ids(self) -> None:
        kb_ids = self.fake_kb_ids()
        doc_map = self.fake_doc_map()
        payloads = {
            item["intentCode"]: item
            for item in build_payloads(
                kb_ids,
                kb_keys_per_leaf(EVAL_SET_PATH, doc_map),
                sample_queries_per_leaf(EVAL_SET_PATH),
            )
        }

        for intent, preferred in PREFERRED_KB_KEYS.items():
            self.assertGreaterEqual(
                len(payloads[intent]["kbIds"]),
                len(set(preferred)),
            )
            self.assertEqual(
                payloads[intent]["kbId"],
                payloads[intent]["kbIds"][0],
            )

    def test_system_nodes_have_customer_service_prompts(self) -> None:
        kb_ids = self.fake_kb_ids()
        doc_map = self.fake_doc_map()
        payloads = {
            item["intentCode"]: item
            for item in build_payloads(
                kb_ids,
                kb_keys_per_leaf(EVAL_SET_PATH, doc_map),
                sample_queries_per_leaf(EVAL_SET_PATH),
            )
        }

        for intent, prompt in SYSTEM_PROMPTS.items():
            self.assertEqual(payloads[intent]["promptTemplate"], prompt)
            self.assertEqual(payloads[intent]["kbIds"], [])
