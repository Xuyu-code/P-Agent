"""Embedding providers for the RAG store.

Two providers, selected by EMBEDDING_PROVIDER:

- "dashscope" (default): remote text-embedding API over the same
  OpenAI-compatible endpoint as the LLM. No local model, no extra download.
- "local": BAAI/bge-small-zh-v1.5 via sentence-transformers. Downloaded once
  (set HF_ENDPOINT=https://hf-mirror.com when HuggingFace is unreachable),
  cached locally afterwards. Requires the optional `sentence-transformers`
  dependency, which is intentionally NOT in requirements.txt (it pulls torch).
"""

from __future__ import annotations

from .. import config


class EmbeddingError(RuntimeError):
    pass


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents with the configured provider."""
    if not texts:
        return []
    if config.EMBEDDING_PROVIDER == "dashscope":
        return _embed_dashscope(texts)
    if config.EMBEDDING_PROVIDER == "local":
        return _embed_local(texts)
    raise EmbeddingError(f"未知 EMBEDDING_PROVIDER: {config.EMBEDDING_PROVIDER!r}（应为 dashscope 或 local）")


def embed_query(text: str) -> list[float]:
    """Embed a retrieval query.

    BGE zh models document an instruction prefix for the query side of
    short-query-to-passage retrieval (documents stay unprefixed). Using it
    materially improves ranking on our mixed Chinese/English corpus.
    """
    if config.EMBEDDING_PROVIDER == "local":
        [vec] = _embed_local([text], query=True)
        return vec
    [vec] = embed_texts([text])
    return vec


def embedding_dim() -> int:
    return config.EMBEDDING_DIM if config.EMBEDDING_PROVIDER == "dashscope" else config.LOCAL_EMBEDDING_DIM


# ---------------------------------------------------------------------------
# DashScope (OpenAI-compatible embeddings endpoint)
# ---------------------------------------------------------------------------

def _embed_dashscope(texts: list[str]) -> list[list[float]]:
    from .llm import get_client  # reuse base_url/api_key/timeout

    resp = get_client().embeddings.create(model=config.EMBEDDING_MODEL, input=texts)
    # OpenAI-compatible endpoints may return items out of order.
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [list(d.embedding) for d in ordered]


# ---------------------------------------------------------------------------
# Local BGE (sentence-transformers, optional dependency)
# ---------------------------------------------------------------------------

_local_model = None

# BGE zh 官方推荐的查询侧指令前缀（文档侧不加）
_BGE_QUERY_PROMPT = "为这个句子生成表示以用于检索相关文章："


def _embed_local(texts: list[str], *, query: bool = False) -> list[list[float]]:
    global _local_model
    if _local_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=local 需要可选依赖 sentence-transformers：\n"
                "  pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple\n"
                "模型可经 ModelScope 下载到本地目录后用 LOCAL_EMBEDDING_MODEL 指向。"
            ) from exc
        _local_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
    prompt = _BGE_QUERY_PROMPT if query else None
    vectors = _local_model.encode(texts, normalize_embeddings=True, prompt=prompt)
    return [list(map(float, v)) for v in vectors]
