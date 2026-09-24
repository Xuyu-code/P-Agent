"""API service layer tests: TestClient + scripted LLM + fake generation service.

No network, no GPU, no real LLM — same mock philosophy as test_e2e_mock.py,
one layer up (HTTP in, HTTP out, SQLite persistence verified).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agent_project import config, llm
from agent_project.api import db, server
from agent_project.graph import nodes
from agent_project.tools import knowledge, puppet_api


def _hit():
    return {
        "doc_id": "01_national_ich_project",
        "snippet": "河湟皮影戏属于青海河湟地区流传的皮影戏地方类型。",
        "score": 0.9,
        "in_scope": True,
        "source": {
            "title": "皮影戏（河湟皮影戏）",
            "publisher": "中国非物质文化遗产网",
            "locator": "x",
            "confidence": "high",
            "claim_scope": "y",
            "license_status": "z",
        },
    }


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # 隔离数据目录（会话库 / 上传目录），并让 db 重新建连
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "data" / "chroma")
    monkeypatch.setattr(config, "UPLOADS_DIR", tmp_path / "data" / "uploads")
    db.reset_for_tests()

    def fake_chat(messages, **kwargs):
        text = messages[-1]["content"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if "意图分类器" in text:
            message = text.split("用户：")[-1]
            if any(k in message for k in ("生成", "头像", "创作")):
                return "create"
            if "历史" in message:
                return "history"
            return "qa"
        if "从对话中提取" in text:
            return '{"role": "女角", "headwear": "凤冠"}'
        if "只能基于给定证据回答" in system:
            return "河湟皮影戏流传于青海河湟地区【1】。"
        return "你好！"

    monkeypatch.setattr(nodes, "llm_chat", fake_chat)
    monkeypatch.setattr(knowledge, "search_heritage_knowledge", lambda *a, **k: [_hit()])
    monkeypatch.setattr(
        puppet_api, "generate_shadow_puppet",
        lambda **kw: {"task_id": "api1", "position": 1},
    )
    monkeypatch.setattr(
        puppet_api, "get_generation_status",
        lambda tid: {"task_id": tid, "status": "done",
                     "result": {"final_url": "http://svc/api/images/api1/final.png",
                                "candidate_urls": ["u1", "u2", "u3"], "elapsed_seconds": 16.0}},
    )
    monkeypatch.setattr(config, "PUPPET_POLL_INTERVAL", 0)

    with TestClient(server.app) as c:
        yield c
    db.reset_for_tests()


# ---------------------------------------------------------------------------
# health / chat / sessions
# ---------------------------------------------------------------------------

def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_creates_session_and_persists(client):
    r1 = client.post("/api/chat", json={"message": "河湟皮影是什么？"})
    assert r1.status_code == 200
    body = r1.json()
    assert body["intent"] == "qa"
    assert "【1】" in body["answer"]
    assert body["citations"][0]["publisher"] == "中国非物质文化遗产网"
    sid = body["session_id"]

    sess = client.get(f"/api/sessions/{sid}").json()
    assert sess["message_count"] == 2  # user + assistant

    r2 = client.post("/api/chat", json={"message": "它是不是国家级非遗？", "session_id": sid})
    assert r2.status_code == 200
    assert r2.json()["session_id"] == sid
    sess = client.get(f"/api/sessions/{sid}").json()
    assert sess["message_count"] == 4
    assert sess["recent_messages"][0]["role"] == "user"


def test_chat_unknown_session_404(client):
    resp = client.post("/api/chat", json={"message": "你好", "session_id": "nope"})
    assert resp.status_code == 404


def test_message_too_long_422(client):
    resp = client.post("/api/chat", json={"message": "x" * 2001})
    assert resp.status_code == 422


def test_same_session_serialized_409(client):
    sid = client.post("/api/chat", json={"message": "你好"}).json()["session_id"]
    lock = server._lock_for(sid)
    lock.acquire()
    try:
        resp = client.post("/api/chat", json={"message": "第二句", "session_id": sid})
        assert resp.status_code == 409
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# lineart upload + creation flow
# ---------------------------------------------------------------------------

def test_lineart_validation(client, tiny_png):
    sid = client.post("/api/chat", json={"message": "你好"}).json()["session_id"]

    bad = client.post(f"/api/sessions/{sid}/lineart",
                      files={"file": ("notes.txt", b"hello", "text/plain")})
    assert bad.status_code == 400

    ok = client.post(f"/api/sessions/{sid}/lineart",
                     files={"file": ("lineart.png", tiny_png.read_bytes(), "image/png")})
    assert ok.status_code == 200
    assert ok.json()["lineart_path"].endswith(".png")
    assert client.get(f"/api/sessions/{sid}").json()["lineart_path"].endswith(".png")


def test_lineart_unknown_session_404(client, tiny_png):
    resp = client.post("/api/sessions/nope/lineart",
                       files={"file": ("a.png", tiny_png.read_bytes(), "image/png")})
    assert resp.status_code == 404


def test_creation_flow_via_api(client, tiny_png):
    sid = client.post("/api/chat", json={"message": "我想创作一个皮影头像"}).json()["session_id"]
    client.post(f"/api/sessions/{sid}/lineart",
                files={"file": ("lineart.png", tiny_png.read_bytes(), "image/png")})
    resp = client.post("/api/chat",
                       json={"message": "帮我生成一个戴凤冠的女角头像，直接生成", "session_id": sid})
    body = resp.json()
    assert body["intent"] == "create"
    assert body["generation"]["final_url"].endswith("final.png")

    sess = client.get(f"/api/sessions/{sid}").json()
    assert sess["slots"]["role"] == "女角"
    assert sess["slots"]["generated_once"] is True


# ---------------------------------------------------------------------------
# token metering
# ---------------------------------------------------------------------------

def test_usage_listener_records_and_tags_session(tmp_path, monkeypatch):
    # 隔离数据目录：本测试直接触发 _record_usage，若进程里残留了
    # db.record_llm_call 监听器（TestClient lifespan 注册过），落库必须落在
    # 临时目录，不得污染真实 data/agent.db（实测踩过这个坑）
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    db.reset_for_tests()
    records = []
    llm.add_usage_listener(records.append)
    llm.add_usage_listener(records.append)  # 幂等：重复注册不得叠加
    try:
        fake_resp = type("R", (), {"usage": type("U", (), {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})()})()
        llm._record_usage("test-model", fake_resp)
        token = llm.set_session_context("s-test")
        llm._record_usage("test-model", fake_resp)
        llm.reset_session_context(token)
    finally:
        llm.remove_usage_listener(records.append)
    assert len(records) == 2  # 两次 _record_usage 各触发一次，重复注册不生效
    assert records[0]["session_id"] is None
    assert records[1]["session_id"] == "s-test"
    assert records[1]["total_tokens"] == 15
    db.reset_for_tests()


def test_stats_aggregation(client):
    db.record_llm_call({"session_id": "s1", "model": "m", "prompt_tokens": 100,
                        "completion_tokens": 50, "total_tokens": 150})
    db.record_llm_call({"session_id": "s1", "model": "m", "prompt_tokens": 200,
                        "completion_tokens": 100, "total_tokens": 300})
    client.post("/api/chat", json={"message": "河湟皮影是什么？"})

    stats = client.get("/api/stats").json()
    assert stats["llm_calls_total"] == 2
    assert stats["tokens_total"] == 450
    assert stats["sessions"] == 1
    assert stats["messages"] == 2
    assert stats["by_intent"][0]["intent"] == "qa"


def test_usage_since_is_session_scoped(client):
    marker = db.max_llm_call_id()
    db.record_llm_call({"session_id": "a", "model": "m", "prompt_tokens": 1,
                        "completion_tokens": 1, "total_tokens": 2})
    db.record_llm_call({"session_id": "b", "model": "m", "prompt_tokens": 10,
                        "completion_tokens": 10, "total_tokens": 20})
    usage = db.usage_since(marker, "a")
    assert usage == {"calls": 1, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


# ---------------------------------------------------------------------------
# SSE 流式端点（2026-08-23 加固轮新增）
# ---------------------------------------------------------------------------

def test_chat_stream_endpoint_sse(client, monkeypatch):
    """QA 意图：SSE 流应先收到逐 token 事件，再以 done 事件收尾（含引用与用量）。"""
    import json as _json

    monkeypatch.setattr(nodes, "llm_chat_stream",
                        lambda messages: iter(["河湟皮影流传于", "青海河湟地区【1】"]))
    with client.stream("POST", "/api/chat/stream",
                       json={"message": "河湟皮影是什么？"}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in resp.iter_lines():
            if line.startswith("data: "):
                events.append(_json.loads(line[6:]))

    tokens = [e for e in events if e["type"] == "token"]
    done = [e for e in events if e["type"] == "done"]
    assert "".join(t["content"] for t in tokens) == "河湟皮影流传于青海河湟地区【1】"
    assert len(done) == 1
    assert done[0]["intent"] == "qa"
    assert done[0]["citations"][0]["publisher"] == "中国非物质文化遗产网"
    assert done[0]["session_id"]
    # 会话持久化：流式与非流式一致落库
    sess = client.get(f"/api/sessions/{done[0]['session_id']}").json()
    assert sess["message_count"] == 2


def test_chat_stream_endpoint_nonqa_no_tokens(client, monkeypatch):
    """非 QA 意图（history，不走 LLM）：无 token 事件，直接 done。"""
    import json as _json

    monkeypatch.setattr(puppet_api, "get_generation_history",
                        lambda limit=5: {"items": [], "total": 0})
    with client.stream("POST", "/api/chat/stream",
                       json={"message": "查一下历史记录"}) as resp:
        events = [_json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    assert not [e for e in events if e["type"] == "token"]
    done = [e for e in events if e["type"] == "done"]
    assert len(done) == 1
    assert done[0]["intent"] == "history"


# ---------------------------------------------------------------------------
# 会话回放端点（Streamlit 客户端模式依赖，2026-08-26）
# ---------------------------------------------------------------------------

def test_create_session_and_messages_replay(client):
    sid = client.post("/api/sessions").json()["session_id"]
    assert sid

    client.post("/api/chat", json={"message": "河湟皮影是什么？", "session_id": sid})
    resp = client.get(f"/api/sessions/{sid}/messages")
    assert resp.status_code == 200
    msgs = resp.json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["citations"][0]["publisher"] == "中国非物质文化遗产网"
    assert msgs[1]["trace"]

    # 空会话回放为空；不存在会话 404
    empty = client.post("/api/sessions").json()["session_id"]
    assert client.get(f"/api/sessions/{empty}/messages").json()["messages"] == []
    assert client.get("/api/sessions/nope/messages").status_code == 404


def test_chat_response_carries_slots(client, tiny_png):
    sid = client.post("/api/sessions").json()["session_id"]
    client.post(f"/api/sessions/{sid}/lineart",
                files={"file": ("a.png", tiny_png.read_bytes(), "image/png")})
    body = client.post("/api/chat",
                       json={"message": "帮我生成一个戴凤冠的女角头像，直接生成", "session_id": sid}).json()
    assert body["slots"]["role"] == "女角"
    assert body["slots"]["generated_once"] is True
