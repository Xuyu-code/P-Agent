"""Chroma vector store: two collections with strict provenance metadata.

- heritage_facts : self-written public-source fact cards (one doc per card).
- project_facts  : our own public service notes, chunked.

The separation is deliberate (see data_sources/SOURCE_CARDS_README.md):
project documents must never be presented as traditional cultural records.
"""

from __future__ import annotations

import chromadb

from .. import config

_client: chromadb.PersistentClient | None = None

# Metadata fields every heritage card must carry (source_index + card README).
REQUIRED_CARD_METADATA = (
    "source_type",
    "title",
    "publisher",
    "source_domain",
    "locator",
    "license_status",
    "license_note",
    "confidence",
    "claim_scope",
)


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        config.ensure_data_dirs()
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return _client


def get_collection(name: str) -> chromadb.Collection:
    return get_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collections() -> None:
    """Drop and recreate both collections (used by `ingest --reset`)."""
    client = get_client()
    for name in (config.COLLECTION_HERITAGE, config.COLLECTION_PROJECT):
        try:
            client.delete_collection(name)
        except (ValueError, chromadb.errors.NotFoundError):
            pass
