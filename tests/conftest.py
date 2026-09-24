"""Shared pytest fixtures: temp Chroma dir, deterministic fake embeddings."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_project import config  # noqa: E402
from agent_project.retrieval import store  # noqa: E402

FAKE_DIM = 32


def fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic md5-based vectors: same text (modulo whitespace) -> same vector."""
    vectors = []
    for text in texts:
        digest = hashlib.md5(text.strip().encode("utf-8")).digest()
        vec = [(b / 255.0) - 0.5 for b in digest] * (FAKE_DIM // 16)
        vectors.append(vec)
    return vectors


@pytest.fixture()
def temp_store(tmp_path, monkeypatch):
    """Point Chroma at a temp dir and use fake embeddings everywhere."""
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    monkeypatch.setattr(store, "_client", None)
    monkeypatch.setattr(
        "agent_project.retrieval.ingest.embed_texts", fake_embed
    )
    monkeypatch.setattr(
        "agent_project.tools.knowledge.embed_query",
        lambda text: fake_embed([text])[0],
    )
    yield tmp_path
    monkeypatch.setattr(store, "_client", None)


@pytest.fixture()
def tiny_png(tmp_path) -> Path:
    """A file that passes the agent's local lineart validation (suffix+size)."""
    path = tmp_path / "lineart.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return path
