"""Tests for the generation-service HTTP tools (mocked transport, no server)."""

from __future__ import annotations

import httpx
import pytest

from agent_project.tools import puppet_api


@pytest.fixture()
def mock_http(monkeypatch):
    """Route puppet_api's httpx.request through an httpx.MockTransport."""
    state = {"handler": lambda request: httpx.Response(200, json={})}

    def handler(request: httpx.Request) -> httpx.Response:
        state["last_request"] = request
        return state["handler"](request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(puppet_api.httpx, "request", client.request)
    return state


class TestGenerate:
    def test_success(self, mock_http, tiny_png):
        mock_http["handler"] = lambda req: httpx.Response(200, json={"task_id": "abc123", "position": 1})
        out = puppet_api.generate_shadow_puppet(prompt="河湟皮影风格头像", lineart_path=str(tiny_png))
        assert out == {"task_id": "abc123", "position": 1}
        body = mock_http["last_request"].content
        assert "河湟皮影风格头像".encode() in body  # prompt actually sent

    def test_rate_limited(self, mock_http, tiny_png):
        mock_http["handler"] = lambda req: httpx.Response(429, json={"detail": "rate limited"})
        with pytest.raises(puppet_api.PuppetAPIError, match="限 10 次"):
            puppet_api.generate_shadow_puppet(prompt="x", lineart_path=str(tiny_png))

    def test_model_not_loaded(self, mock_http, tiny_png):
        mock_http["handler"] = lambda req: httpx.Response(503, json={"detail": "model not loaded"})
        with pytest.raises(puppet_api.PuppetAPIError, match="模型尚未加载"):
            puppet_api.generate_shadow_puppet(prompt="x", lineart_path=str(tiny_png))

    def test_service_down(self, mock_http, tiny_png):
        def boom(req):
            raise httpx.ConnectError("refused")

        mock_http["handler"] = boom
        with pytest.raises(puppet_api.PuppetAPIError, match="无法连接皮影生成服务"):
            puppet_api.generate_shadow_puppet(prompt="x", lineart_path=str(tiny_png))

    def test_local_validation_never_touches_http(self, mock_http, tmp_path):
        def fail(req):  # any HTTP attempt is a test failure
            raise AssertionError("HTTP should not be called on invalid local input")

        mock_http["handler"] = fail
        with pytest.raises(puppet_api.PuppetAPIError, match="提示词不能为空"):
            puppet_api.generate_shadow_puppet(prompt="  ", lineart_path="whatever.png")
        with pytest.raises(puppet_api.PuppetAPIError, match="1～8"):
            puppet_api.generate_shadow_puppet(prompt="x", lineart_path="whatever.png", k=0)
        with pytest.raises(puppet_api.PuppetAPIError, match="线稿文件不存在"):
            puppet_api.generate_shadow_puppet(prompt="x", lineart_path=str(tmp_path / "nope.png"))
        bad = tmp_path / "lineart.txt"
        bad.write_text("not an image")
        with pytest.raises(puppet_api.PuppetAPIError, match="格式不支持"):
            puppet_api.generate_shadow_puppet(prompt="x", lineart_path=str(bad))


class TestStatusAndHistory:
    def test_status_urls_become_absolute(self, mock_http):
        mock_http["handler"] = lambda req: httpx.Response(
            200,
            json={
                "task_id": "t1",
                "status": "done",
                "position": None,
                "created_at": 1.0,
                "result": {
                    "candidate_urls": ["/api/images/t1/candidate_1.png"],
                    "final_url": "/api/images/t1/final.png",
                    "selected_url": "/api/images/t1/selected.png",
                },
            },
        )
        payload = puppet_api.get_generation_status("t1")
        assert payload["result"]["final_url"] == f"{puppet_api.config.PUPPET_API_BASE}/api/images/t1/final.png"
        assert payload["result"]["candidate_urls"][0].startswith("http")

    def test_history_clamps_and_absolutizes(self, mock_http):
        mock_http["handler"] = lambda req: httpx.Response(
            200,
            json={
                "total": 1,
                "items": [{"task_id": "t1", "status": "done", "thumbnail_url": "/api/images/t1/final.png"}],
            },
        )
        payload = puppet_api.get_generation_history(limit=9999)
        assert payload["items"][0]["thumbnail_url"].startswith("http")

    def test_unknown_task(self, mock_http):
        mock_http["handler"] = lambda req: httpx.Response(404, json={"detail": "Unknown task_id."})
        with pytest.raises(puppet_api.PuppetAPIError, match="未找到"):
            puppet_api.get_generation_status("missing")
