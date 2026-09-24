"""Build the RAG index from data_sources/.

Heritage cards: parsed as (front-matter-ish `- key: value` metadata + body),
stored one document per card, all REQUIRED_CARD_METADATA fields asserted.
Project service notes: markdown chunked by headers with size limits.

Only files inside data_sources/ are indexed — the ingest never fetches the
original web pages (internal_paraphrase_only policy: no full-text copying).

Usage:
    python -m agent_project.retrieval.ingest [--reset]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .. import config
from . import store
from .embeddings import embed_texts

_META_LINE_RE = re.compile(r"^-\s*([a-z_]+):\s*(.+?)\s*$")

# ---------------------------------------------------------------------------
# Heritage cards
# ---------------------------------------------------------------------------

def parse_card(path: Path) -> tuple[dict[str, str], str]:
    """Split a card file into (metadata, document text)."""
    text = path.read_text(encoding="utf-8")
    meta: dict[str, str] = {}
    for line in text.splitlines():
        m = _META_LINE_RE.match(line)
        if m:
            meta[m.group(1)] = m.group(2)
    missing = [k for k in store.REQUIRED_CARD_METADATA if k not in meta]
    if missing:
        raise ValueError(f"{path.name} 缺少元数据字段: {', '.join(missing)}")
    meta["source_file"] = path.name
    # The whole card is self-written (summary + usage hints), so the full
    # body is safe to embed; the title line keeps the topic prominent.
    return meta, text


def load_heritage_cards(cards_dir: Path = config.HERITAGE_CARDS_DIR) -> tuple[list[str], list[str], list[dict]]:
    ids, docs, metas = [], [], []
    for path in sorted(cards_dir.glob("*.md")):
        meta, doc = parse_card(path)
        ids.append(path.stem)
        docs.append(doc)
        metas.append(meta)
    return ids, docs, metas


# ---------------------------------------------------------------------------
# Project facts (own documents, chunked)
# ---------------------------------------------------------------------------

def chunk_markdown(text: str, *, size: int = config.PROJECT_CHUNK_SIZE, overlap: int = config.PROJECT_CHUNK_OVERLAP) -> list[str]:
    """One chunk per markdown section; split only oversized sections.

    Sections are never packed together: mixing unrelated topics into one chunk
    dilutes its embedding, so each heading remains an independent retrieval unit.
    """
    parts = re.split(r"(?m)^(#{1,3}\s.*)$", text)
    units: list[str] = []
    if parts[0].strip():
        units.append(parts[0].strip())
    for i in range(1, len(parts) - 1, 2):
        header, body = parts[i], parts[i + 1]
        units.append((header + "\n" + body.strip()).strip())

    chunks: list[str] = []
    for unit in units:
        while len(unit) > size:  # oversized section: hard-split with overlap
            chunks.append(unit[:size])
            unit = unit[size - overlap:]
        if unit.strip():
            chunks.append(unit)
    return chunks


def load_project_facts(facts_dir: Path = config.PROJECT_FACTS_DIR) -> tuple[list[str], list[str], list[dict]]:
    ids, docs, metas = [], [], []
    for path in sorted(facts_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for idx, chunk in enumerate(chunk_markdown(text)):
            ids.append(f"{path.stem}#{idx:03d}")
            docs.append(chunk)
            metas.append(
                {
                    "source_type": "project_document",
                    "title": path.name,
                    "publisher": "hehuang-shadow-puppet project",
                    "source_domain": "local",
                    "locator": f"data_sources/project_facts/{path.name}",
                    "license_status": "author_owned",
                    "license_note": "项目自有服务说明；回答时须与传统文化史料区分。",
                    "confidence": "high",
                    "claim_scope": "本项目的公开功能与服务接口说明",
                    "source_file": path.name,
                }
            )
    return ids, docs, metas


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------

def _add_in_batches(collection, ids, docs, metas, batch_size: int = 16) -> int:
    added = 0
    for start in range(0, len(ids), batch_size):
        batch_docs = docs[start:start + batch_size]
        collection.add(
            ids=ids[start:start + batch_size],
            documents=batch_docs,
            metadatas=metas[start:start + batch_size],
            embeddings=embed_texts(batch_docs),
        )
        added += len(batch_docs)
    return added


def run_ingest(reset: bool = False) -> dict[str, int]:
    if reset:
        store.reset_collections()
    counts = {}
    for name, loader in (
        (config.COLLECTION_HERITAGE, load_heritage_cards),
        (config.COLLECTION_PROJECT, load_project_facts),
    ):
        ids, docs, metas = loader()
        collection = store.get_collection(name)
        counts[name] = _add_in_batches(collection, ids, docs, metas)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG index from data_sources/.")
    parser.add_argument("--reset", action="store_true", help="drop existing collections first")
    args = parser.parse_args()
    counts = run_ingest(reset=args.reset)
    for name, n in counts.items():
        print(f"[ingest] {name}: {n} documents")


if __name__ == "__main__":
    main()
