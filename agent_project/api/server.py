"""FastAPI service layer for the agent (default port 8001).

Same LangGraph graph as the Streamlit UI — this layer turns the demo into a
service: sessions and messages persist in SQLite across restarts, every LLM
call is metered into a cost table, and any frontend (the future Vue 集成,
scripts, third parties) can drive the agent over HTTP.

Concurrency model: the graph is synchronous and creation turns poll the
generation service for tens of seconds, so `chat` offloads `graph.invoke` to
a threadpool (event loop stays responsive); the downstream GPU service is a
serial queue anyway. A per-session lock serializes double-clicks on the same
conversation (retry-safe without client-side idempotency keys).
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from .. import config, llm
from ..graph import nodes
from ..graph.builder import get_graph
from . import db

_LINEART_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_LINEART_MAX_BYTES = 5 * 1024 * 1024  # 与 create_node 对用户的承诺一致
_CONTEXT_MESSAGES = 6  # 与 Streamlit 端一致：最近 3 个问答对

# ---------------------------------------------------------------------------
# Per-session serialization (double-click / retry safety)
# ---------------------------------------------------------------------------

_session_locks: dict[str, threading.Lock] = {}
_session_locks_guard = threading.Lock()


def _lock_for(session_id: str) -> threading.Lock:
    with _session_locks_guard:
        return _session_locks.setdefault(session_id, threading.Lock())


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_data_dirs()
    llm.add_usage_listener(db.record_llm_call)  # 每次 LLM 调用落库（幂等注册）
    yield
    llm.remove_usage_listener(db.record_llm_call)


app = FastAPI(
    title="河湟非遗知识问答与皮影创作 Agent API",
    version="0.2.0",
    lifespan=lifespan,
)

_origins = os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:8000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    answer: str
    citations: list[dict]
    generation: dict
    trace: list[dict]
    latency_s: float
    usage: dict  # 本轮 LLM token 用量（来自 llm_calls 落库，跨并发不串账）
    slots: dict | None = None  # 当前创作槽位（供客户端侧栏同步展示）


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"name": "hehuang-heritage-agent", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def health():
    puppet_ok = False
    try:
        resp = httpx.get(f"{config.PUPPET_API_BASE}/api/health", timeout=3.0)
        puppet_ok = resp.status_code == 200
    except Exception:
        pass
    return {
        "status": "ok",
        "llm_model": config.LLM_MODEL,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "puppet_api": config.PUPPET_API_BASE,
        "puppet_api_reachable": puppet_ok,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if req.session_id:
        session = db.get_session(req.session_id)
        if session is None:
            raise HTTPException(404, f"会话不存在: {req.session_id}")
    else:
        session = db.get_session(db.create_session())
    sid = session["session_id"]

    lock = _lock_for(sid)
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "该会话有正在处理的请求（创作生成可能需数十秒），请稍候再发")
    try:
        history = [
            HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
            for m in db.get_recent_messages(sid, _CONTEXT_MESSAGES)
        ]
        history.append(HumanMessage(content=req.message))
        db.add_message(sid, "user", req.message)

        marker = db.max_llm_call_id()
        token = llm.set_session_context(sid)
        started = time.perf_counter()
        try:
            result = await run_in_threadpool(
                get_graph().invoke,
                {
                    "messages": history,
                    "slots": session["slots"] or None,
                    "lineart_path": session["lineart_path"],
                },
            )
        finally:
            llm.reset_session_context(token)
        latency = time.perf_counter() - started

        answer = result.get("answer") or "（没有生成回答）"
        intent = result.get("intent", "chat")
        citations = result.get("citations") or []
        generation = result.get("generation") or {}
        trace = result.get("trace") or []
        db.add_message(
            sid, "assistant", answer,
            intent=intent, trace=trace, citations=citations,
            generation=generation, latency_s=round(latency, 1),
        )
        if result.get("slots") is not None:
            db.update_session(sid, slots=result["slots"])

        return ChatResponse(
            session_id=sid, intent=intent, answer=answer, citations=citations,
            generation=generation, trace=trace, latency_s=round(latency, 1),
            usage=db.usage_since(marker, sid),
            slots=result.get("slots"),
        )
    finally:
        lock.release()


# 流式请求的串行锁：模块级 sink 是共享通道，且 Kimi 3 RPM 使并发流式本就无法真正并行
# （与下游单 GPU 串行队列同理——约束驱动，而非偷懒）
_stream_invoke_lock = threading.Lock()


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式对话：QA 回答逐 token 推送（type=token），完成后发完整结果（type=done）。

    非 QA 意图（创作/历史/闲聊）不产 token 流，直接收到 done 事件——调用方
    按 type 分支处理即可。事件格式：data: {"type": ..., ...}\\n\\n
    """
    if req.session_id:
        session = db.get_session(req.session_id)
        if session is None:
            raise HTTPException(404, f"会话不存在: {req.session_id}")
    else:
        session = db.get_session(db.create_session())
    sid = session["session_id"]

    session_lock = _lock_for(sid)
    if not session_lock.acquire(blocking=False):
        raise HTTPException(409, "该会话有正在处理的请求（创作生成可能需数十秒），请稍候再发")

    async def event_gen():
        token_q: queue.Queue = queue.Queue()
        holder: dict = {}
        marker = db.max_llm_call_id()
        token_ctx = llm.set_session_context(sid)
        started = time.perf_counter()

        def work():
            try:
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                    for m in db.get_recent_messages(sid, _CONTEXT_MESSAGES)
                ]
                history.append(HumanMessage(content=req.message))
                db.add_message(sid, "user", req.message)
                nodes.set_stream_sink(token_q)
                holder["result"] = get_graph().invoke(
                    {
                        "messages": history,
                        "slots": session["slots"] or None,
                        "lineart_path": session["lineart_path"],
                    }
                )
            except Exception as exc:
                holder["error"] = exc
            finally:
                nodes.clear_stream_sink()
                token_q.put(None)

        await asyncio.to_thread(_stream_invoke_lock.acquire)
        try:
            task = asyncio.get_running_loop().run_in_executor(None, work)
            while True:
                tok = await asyncio.to_thread(token_q.get)
                if tok is None:
                    break
                yield f"data: {json.dumps({'type': 'token', 'content': tok}, ensure_ascii=False)}\n\n"
            await task
        finally:
            llm.reset_session_context(token_ctx)
            _stream_invoke_lock.release()
            session_lock.release()

        if "error" in holder:
            yield f"data: {json.dumps({'type': 'error', 'message': str(holder['error'])}, ensure_ascii=False)}\n\n"
            return

        result = holder["result"]
        latency = time.perf_counter() - started
        answer = result.get("answer") or "（没有生成回答）"
        intent = result.get("intent", "chat")
        citations = result.get("citations") or []
        generation = result.get("generation") or {}
        trace = result.get("trace") or []
        db.add_message(
            sid, "assistant", answer,
            intent=intent, trace=trace, citations=citations,
            generation=generation, latency_s=round(latency, 1),
        )
        if result.get("slots") is not None:
            db.update_session(sid, slots=result["slots"])
        done = {
            "type": "done", "session_id": sid, "intent": intent, "answer": answer,
            "citations": citations, "generation": generation, "trace": trace,
            "latency_s": round(latency, 1), "usage": db.usage_since(marker, sid),
            "slots": result.get("slots"),
        }
        yield f"data: {json.dumps(done, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/api/sessions/{session_id}/messages")
def get_session_messages(session_id: str, limit: int = 50):
    """完整会话消息（含引用/生成/trace/耗时）——前端刷新后回放用。"""
    if db.get_session(session_id) is None:
        raise HTTPException(404, f"会话不存在: {session_id}")
    return {"session_id": session_id, "messages": db.get_messages_full(session_id, limit)}


@app.post("/api/sessions")
def create_empty_session():
    """创建一个空会话（前端开局用，不必等到首条消息）。"""
    return {"session_id": db.create_session()}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = db.get_session(session_id)
    if session is None:
        raise HTTPException(404, f"会话不存在: {session_id}")
    return {
        **session,
        "message_count": db.count_messages(session_id),
        "recent_messages": db.get_recent_messages(session_id, 20),
    }


@app.post("/api/sessions/{session_id}/lineart")
async def upload_lineart(session_id: str, file: UploadFile):
    if db.get_session(session_id) is None:
        raise HTTPException(404, f"会话不存在: {session_id}")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _LINEART_SUFFIXES:
        raise HTTPException(400, f"不支持的格式 {suffix!r}：仅接受 {'/'.join(sorted(_LINEART_SUFFIXES))}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "文件为空")
    if len(data) > _LINEART_MAX_BYTES:
        raise HTTPException(400, f"文件 {len(data)/1048576:.1f}MB 超过 5MB 上限")
    config.ensure_data_dirs()
    saved = config.UPLOADS_DIR / f"{session_id}_{uuid.uuid4().hex[:8]}{suffix}"
    saved.write_bytes(data)
    db.update_session(session_id, lineart_path=str(saved))
    return {"session_id": session_id, "lineart_path": str(saved), "size_bytes": len(data)}


@app.get("/api/stats")
def get_stats():
    return db.stats()
