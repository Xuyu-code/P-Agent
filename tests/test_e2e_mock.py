"""End-to-end tests through the compiled graph with LLM + tools mocked.

Covers the two flagship flows without any GPU, network, or API key:
1. two-turn creation (ask for lineart -> generate after upload);
2. evidence-bounded QA (hit and no-hit).
"""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from agent_project import config
from agent_project.graph import builder, nodes
from agent_project.tools import knowledge, puppet_api


@pytest.fixture()
def scripted(monkeypatch):
    """Wire a deterministic LLM + generation service into the graph."""
    def fake_chat(messages, **kwargs):
        text = messages[-1]["content"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if "意图分类器" in text:
            # 模板本身含“生成/头像”字样，只对“用户：”之后的真实输入分流
            message = text.split("用户：")[-1]
            if any(k in message for k in ("生成", "头像", "线稿", "直接生成")):
                return "create"
            if "历史" in message:
                return "history"
            return "qa"
        if "从对话中提取" in text:
            return '{"role": "女角", "facing": "右侧面", "headwear": "凤冠", "color_palette": "暖黄色"}'
        if "只能基于给定证据回答" in system:
            return "河湟皮影戏是青海河湟地区流传的皮影戏地方类型【1】。"
        return "你好！"

    monkeypatch.setattr(nodes, "llm_chat", fake_chat)
    monkeypatch.setattr(
        puppet_api, "generate_shadow_puppet",
        lambda **kw: {"task_id": "e2e1", "position": 1},
    )
    monkeypatch.setattr(
        puppet_api, "get_generation_status",
        lambda tid: {"task_id": tid, "status": "done",
                     "result": {"final_url": "http://svc/api/images/e2e1/final.png",
                                "candidate_urls": ["u1", "u2", "u3"], "elapsed_seconds": 17.2}},
    )
    monkeypatch.setattr(config, "PUPPET_POLL_INTERVAL", 0)
    monkeypatch.setattr(
        knowledge, "search_heritage_knowledge",
        lambda query, **kw: [{
            "doc_id": "01_national_ich_project", "snippet": "河湟皮影戏属于青海河湟地区流传的皮影戏地方类型。",
            "score": 0.95, "in_scope": True,
            "source": {"title": "皮影戏（河湟皮影戏）", "publisher": "中国非物质文化遗产网",
                       "locator": "https://www.ihchina.cn/project_details/13418.html",
                       "confidence": "high", "claim_scope": "项目身份", "license_status": "internal_paraphrase_only"},
        }],
    )
    return builder.build_graph()


class TestCreationFlow:
    def test_two_turn_creation(self, scripted, tiny_png):
        graph = scripted
        # Turn 1: no lineart yet -> must ask for upload, no fake success.
        turn1 = graph.invoke({"messages": [HumanMessage(content="帮我生成一个戴凤冠的女角头像")]})
        assert turn1["intent"] == "create"
        assert "上传" in turn1["answer"]
        assert "generation" not in turn1

        # Turn 2: lineart uploaded; slots from turn 1 are carried over.
        turn2 = graph.invoke(
            {
                "messages": [HumanMessage(content="线稿传好了，右侧面、暖黄色")],
                "slots": turn1["slots"],
                "lineart_path": str(tiny_png),
            }
        )
        assert "生成完成" in turn2["answer"]
        assert turn2["generation"]["final_url"].endswith("/api/images/e2e1/final.png")
        # slot merging: 凤冠 from turn 1 survived into the prompt
        assert "凤冠" in turn2["generation"]["prompt"]
        assert "右侧面" in turn2["generation"]["prompt"]


class TestQaFlow:
    def test_qa_with_citation(self, scripted):
        out = scripted.invoke({"messages": [HumanMessage(content="河湟皮影是什么？")]})
        assert out["intent"] == "qa"
        assert "【1】" in out["answer"]
        assert out["citations"][0]["publisher"] == "中国非物质文化遗产网"

    def test_qa_without_evidence_sets_boundary(self, scripted, monkeypatch):
        monkeypatch.setattr(knowledge, "search_heritage_knowledge", lambda *a, **k: [])
        out = scripted.invoke({"messages": [HumanMessage(content="凤冠在河湟皮影里象征什么？")]})
        assert "不能凭空补全" in out["answer"]
        assert out["citations"] == []
