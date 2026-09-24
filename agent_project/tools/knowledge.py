"""Knowledge-retrieval tools over the Chroma RAG store.

Every hit carries its provenance metadata (title / publisher / locator /
confidence / claim_scope) so answers can cite sources and the QA node can
decide whether the evidence actually covers the question.
"""

from __future__ import annotations

from typing import Any

from .. import config
from ..retrieval import store
from ..retrieval.embeddings import embed_query

_VALID_COLLECTIONS = {config.COLLECTION_HERITAGE, config.COLLECTION_PROJECT}


class KnowledgeError(RuntimeError):
    pass


def _distance_to_score(distance: float | None) -> float | None:
    """Chroma cosine distance -> rough similarity in [0, 1] for display."""
    if distance is None:
        return None
    return round(max(0.0, min(1.0, 1.0 - distance)), 4)


def search_heritage_knowledge(
    query: str,
    top_k: int = config.RETRIEVAL_TOP_K,
    collection: str = config.COLLECTION_HERITAGE,
) -> list[dict[str, Any]]:
    """Semantic search over a knowledge collection.

    Returns a list of {doc_id, snippet, score, in_scope, source:{...}}.
    `in_scope=False` marks hits beyond the collection's distance threshold
    (the two collections use separate thresholds because their document
    distributions differ): the caller should treat them as
    "no usable evidence", not as weak facts.
    """
    if not query or not query.strip():
        raise KnowledgeError("检索 query 不能为空。")
    if collection not in _VALID_COLLECTIONS:
        raise KnowledgeError(f"未知知识库 collection: {collection!r}")
    top_k = max(1, min(int(top_k), 20))
    max_distance = (
        config.RETRIEVAL_MAX_DISTANCE_PROJECT
        if collection == config.COLLECTION_PROJECT
        else config.RETRIEVAL_MAX_DISTANCE
    )

    col = store.get_collection(collection)
    if col.count() == 0:
        return []
    query_vec = embed_query(query.strip())
    res = col.query(query_embeddings=[query_vec], n_results=min(top_k, col.count()))

    hits = []
    for doc_id, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append(
            {
                "doc_id": doc_id,
                "snippet": doc,
                "score": _distance_to_score(dist),
                "in_scope": dist <= max_distance,
                "source": {
                    "title": meta.get("title"),
                    "publisher": meta.get("publisher"),
                    "locator": meta.get("locator"),
                    "confidence": meta.get("confidence"),
                    "claim_scope": meta.get("claim_scope"),
                    "license_status": meta.get("license_status"),
                },
            }
        )
    return hits


def get_heritage_evidence(
    document_ids: list[str],
    collection: str = config.COLLECTION_HERITAGE,
) -> list[dict[str, Any]]:
    """Fetch full card texts by id — guards against quoting snippets out of context."""
    if not document_ids:
        raise KnowledgeError("document_ids 不能为空。")
    if collection not in _VALID_COLLECTIONS:
        raise KnowledgeError(f"未知知识库 collection: {collection!r}")

    col = store.get_collection(collection)
    res = col.get(ids=list(document_ids))
    out = []
    for doc_id, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        out.append(
            {
                "doc_id": doc_id,
                "full_passage": doc,
                "source": {
                    "title": meta.get("title"),
                    "publisher": meta.get("publisher"),
                    "locator": meta.get("locator"),
                    "confidence": meta.get("confidence"),
                    "claim_scope": meta.get("claim_scope"),
                    "license_status": meta.get("license_status"),
                },
            }
        )
    missing = set(document_ids) - set(res["ids"])
    if missing:
        raise KnowledgeError(f"以下 document_id 不存在: {', '.join(sorted(missing))}")
    return out
