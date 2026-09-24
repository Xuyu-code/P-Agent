"""SQLite persistence for the API service layer.

Three tables, all in `config.DATA_DIR/agent.db` (WAL mode):

- sessions   : one row per conversation (slots/lineart survive restarts —
               the Streamlit UI keeps these in memory and loses them on rerun)
- messages   : every user/assistant turn with intent, trace, citations,
               generation payload and latency
- llm_calls  : one row per LLM call (written via llm.add_usage_listener),
               the basis of the /api/stats cost dashboard

Concurrency: a single connection guarded by a lock. The generation service
downstream is a serial single-GPU queue, so request concurrency here is low
by design; correctness first, no premature pooling.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from .. import config

# RLock（可重入）：_connect() 建连时持锁，调用方也持锁，普通 Lock 会死锁
_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    slots_json   TEXT NOT NULL DEFAULT '{}',
    lineart_path TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id),
    role            TEXT NOT NULL,               -- user | assistant
    content         TEXT NOT NULL,
    intent          TEXT,
    trace_json      TEXT,
    citations_json  TEXT,
    generation_json TEXT,
    latency_s       REAL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        TEXT,
    model             TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_tokens      INTEGER,
    created_at        TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        config.ensure_data_dirs()
        _conn = sqlite3.connect(config.DATA_DIR / "agent.db", check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        with _lock:
            _conn.executescript(_SCHEMA)
    return _conn


def reset_for_tests() -> None:
    """Drop the cached connection so a monkeypatched DATA_DIR takes effect."""
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
        _conn = None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session() -> str:
    session_id = uuid.uuid4().hex[:16]
    with _lock:
        _connect().execute(
            "INSERT INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, _now(), _now()),
        )
        _conn.commit()
    return session_id


def get_session(session_id: str) -> dict | None:
    with _lock:
        row = _connect().execute(
            "SELECT session_id, created_at, updated_at, slots_json, lineart_path "
            "FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "session_id": row[0],
        "created_at": row[1],
        "updated_at": row[2],
        "slots": json.loads(row[3]),
        "lineart_path": row[4],
    }


def update_session(session_id: str, *, slots: dict | None = None, lineart_path: str | None = None) -> None:
    current = get_session(session_id)
    if current is None:
        raise KeyError(f"session 不存在: {session_id}")
    new_slots = current["slots"] if slots is None else slots
    new_lineart = current["lineart_path"] if lineart_path is None else lineart_path
    with _lock:
        _connect().execute(
            "UPDATE sessions SET slots_json = ?, lineart_path = ?, updated_at = ? WHERE session_id = ?",
            (json.dumps(new_slots, ensure_ascii=False), new_lineart, _now(), session_id),
        )
        _conn.commit()


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def add_message(
    session_id: str,
    role: str,
    content: str,
    *,
    intent: str | None = None,
    trace: list | None = None,
    citations: list | None = None,
    generation: dict | None = None,
    latency_s: float | None = None,
) -> int:
    with _lock:
        cur = _connect().execute(
            "INSERT INTO messages (session_id, role, content, intent, trace_json, "
            "citations_json, generation_json, latency_s, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, role, content, intent,
                json.dumps(trace, ensure_ascii=False) if trace else None,
                json.dumps(citations, ensure_ascii=False) if citations else None,
                json.dumps(generation, ensure_ascii=False) if generation else None,
                latency_s, _now(),
            ),
        )
        _conn.commit()
        return int(cur.lastrowid)


def get_recent_messages(session_id: str, limit: int = 6) -> list[dict]:
    """Most recent `limit` messages, chronological order (mirrors the 3-pair
    context window the Streamlit UI feeds to the graph)."""
    with _lock:
        rows = _connect().execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def get_messages_full(session_id: str, limit: int = 50) -> list[dict]:
    """完整消息记录（含引用/生成/trace/耗时），供前端刷新后回放整个会话。"""
    with _lock:
        rows = _connect().execute(
            "SELECT role, content, intent, trace_json, citations_json, generation_json, "
            "latency_s, created_at FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    out = []
    for role, content, intent, trace_j, cit_j, gen_j, lat, created in reversed(rows):
        out.append({
            "role": role,
            "content": content,
            "intent": intent,
            "trace": json.loads(trace_j) if trace_j else [],
            "citations": json.loads(cit_j) if cit_j else [],
            "generation": json.loads(gen_j) if gen_j else {},
            "latency_s": lat,
            "created_at": created,
        })
    return out


def count_messages(session_id: str) -> int:
    with _lock:
        (n,) = _connect().execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()
    return int(n)


# ---------------------------------------------------------------------------
# LLM call metering
# ---------------------------------------------------------------------------

def record_llm_call(record: dict) -> None:
    """Listener registered with llm.add_usage_listener — must stay fast and
    never raise (llm swallows listener errors, but keep it cheap anyway)."""
    with _lock:
        _connect().execute(
            "INSERT INTO llm_calls (session_id, model, prompt_tokens, completion_tokens, "
            "total_tokens, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.get("session_id"), record.get("model"),
                record.get("prompt_tokens"), record.get("completion_tokens"),
                record.get("total_tokens"), _now(),
            ),
        )
        _conn.commit()


def max_llm_call_id() -> int:
    with _lock:
        (n,) = _connect().execute("SELECT COALESCE(MAX(id), 0) FROM llm_calls").fetchone()
    return int(n)


def usage_since(row_id: int, session_id: str) -> dict:
    """Race-free per-turn usage: rows for this session written after `row_id`."""
    with _lock:
        row = _connect().execute(
            "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
            "COALESCE(SUM(total_tokens),0) FROM llm_calls WHERE id > ? AND session_id = ?",
            (row_id, session_id),
        ).fetchone()
    return {"calls": row[0], "prompt_tokens": row[1], "completion_tokens": row[2], "total_tokens": row[3]}


def stats() -> dict:
    with _lock:
        conn = _connect()
        totals = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), "
            "COALESCE(SUM(total_tokens),0) FROM llm_calls"
        ).fetchone()
        today = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(total_tokens),0) FROM llm_calls "
            "WHERE created_at >= ?",
            (datetime.now(timezone.utc).date().isoformat(),),
        ).fetchone()
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        by_intent = conn.execute(
            "SELECT intent, COUNT(*), ROUND(AVG(latency_s),1) FROM messages "
            "WHERE role='assistant' AND intent IS NOT NULL GROUP BY intent"
        ).fetchall()
    return {
        "llm_calls_total": totals[0],
        "prompt_tokens_total": totals[1],
        "completion_tokens_total": totals[2],
        "tokens_total": totals[3],
        "llm_calls_today": today[0],
        "tokens_today": today[1],
        "sessions": sessions,
        "messages": messages,
        "by_intent": [
            {"intent": r[0], "turns": r[1], "avg_latency_s": r[2]} for r in by_intent
        ],
    }
