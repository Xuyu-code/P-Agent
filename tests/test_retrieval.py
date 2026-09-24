"""Tests for ingest + retrieval: metadata integrity, chunking, in-scope flags."""

from __future__ import annotations

import pytest

from agent_project import config
from agent_project.retrieval import ingest, store
from agent_project.tools import knowledge


class TestCardParsing:
    def test_all_real_cards_have_required_metadata(self):
        ids, docs, metas = ingest.load_heritage_cards()
        assert len(ids) == 10  # the ten heritage fact cards
        for meta in metas:
            for field in store.REQUIRED_CARD_METADATA:
                assert meta.get(field), f"{meta.get('source_file')} missing {field}"

    def test_card_doc_is_single_chunk_with_title(self):
        ids, docs, metas = ingest.load_heritage_cards()
        assert ids[0] == "01_national_ich_project"
        assert docs[0].startswith("# 河湟皮影戏")


class TestChunking:
    def test_headers_stay_with_body(self):
        text = "intro\n\n## 第一节\n内容A\n\n## 第二节\n内容B"
        chunks = ingest.chunk_markdown(text, size=800, overlap=120)
        joined = "\n".join(chunks)
        assert "## 第一节" in joined and "内容A" in joined

    def test_oversized_section_is_hard_split_with_overlap(self):
        text = "## 长章节\n" + "字" * 2000
        chunks = ingest.chunk_markdown(text, size=800, overlap=120)
        assert len(chunks) >= 3
        assert all(len(c) <= 800 for c in chunks)


class TestIngestAndSearch:
    def test_ingest_real_sources(self, temp_store):
        counts = ingest.run_ingest(reset=True)
        assert counts[config.COLLECTION_HERITAGE] == 10
        assert counts[config.COLLECTION_PROJECT] >= 1  # public service overview

        col = store.get_collection(config.COLLECTION_HERITAGE)
        res = col.get(ids=["01_national_ich_project"])
        meta = res["metadatas"][0]
        assert meta["license_status"] == "internal_paraphrase_only"
        assert meta["source_domain"] == "ihchina.cn"

    def test_exact_text_retrieves_itself_in_scope(self, temp_store):
        ingest.run_ingest(reset=True)
        ids, docs, _ = ingest.load_heritage_cards()
        hits = knowledge.search_heritage_knowledge(docs[3], top_k=1)
        assert hits[0]["doc_id"] == ids[3]  # md5 vectors: identical text, distance 0
        assert hits[0]["in_scope"] is True
        assert hits[0]["source"]["title"]

    def test_distance_threshold_marks_hits_out_of_scope(self, temp_store, monkeypatch):
        ingest.run_ingest(reset=True)
        monkeypatch.setattr(config, "RETRIEVAL_MAX_DISTANCE", -1.0)
        hits = knowledge.search_heritage_knowledge("任意问题", top_k=3)
        assert hits and all(h["in_scope"] is False for h in hits)

    def test_empty_collection_returns_no_hits(self, temp_store):
        store.get_collection(config.COLLECTION_HERITAGE)  # create but don't fill
        assert knowledge.search_heritage_knowledge("河湟皮影") == []

    def test_get_evidence_full_passage_and_missing_ids(self, temp_store):
        ingest.run_ingest(reset=True)
        evidence = knowledge.get_heritage_evidence(["09_zhou_banghui_inheritor"])
        assert "周邦辉" in evidence[0]["full_passage"]
        assert evidence[0]["source"]["locator"].startswith("https://")
        with pytest.raises(knowledge.KnowledgeError, match="不存在"):
            knowledge.get_heritage_evidence(["no_such_card"])

    def test_project_facts_labelled_as_project_documents(self, temp_store):
        ingest.run_ingest(reset=True)
        col = store.get_collection(config.COLLECTION_PROJECT)
        res = col.get()
        assert res["metadatas"]
        assert all(m["source_type"] == "project_document" for m in res["metadatas"])
        assert all(m["license_status"] == "author_owned" for m in res["metadatas"])
