"""Tests for graph nodes: intent routing, evidence-bounded QA, creation flow.

The LLM (nodes.llm_chat) and both tool modules are faked; these tests assert
orchestration behavior — routing, boundary answers, slot merging, error
surfacing — not model quality.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_project import config
from agent_project.graph import nodes
from agent_project.schemas.creation import CreationSlots, build_prompt
from agent_project.tools import knowledge, puppet_api


@pytest.fixture()
def fake_llm(monkeypatch):
    """Scripted LLM: responds by prompt shape; records every call."""
    calls = []

    def fake_chat(messages, **kwargs):
        calls.append(messages)
        text = messages[-1]["content"]
        system = messages[0]["content"] if messages[0]["role"] == "system" else ""
        if "意图分类器" in text:
            return state.intent
        if "从对话中提取" in text:
            return state.extract_json
        if "只能基于给定证据回答" in system:
            return state.qa_answer
        return "闲聊回答"

    class State:
        intent = "chat"
        extract_json = "{}"
        qa_answer = "带引用的回答【1】"

    state = State()
    monkeypatch.setattr(nodes, "llm_chat", fake_chat)
    state.calls = calls
    return state


def _hit(doc_id="01_national_ich_project", in_scope=True, title="河湟皮影戏：国家级非遗项目条目"):
    return {
        "doc_id": doc_id,
        "snippet": "河湟皮影戏属于青海河湟地区流传的皮影戏地方类型。",
        "score": 0.9 if in_scope else 0.2,
        "in_scope": in_scope,
        "source": {"title": title, "publisher": "中国非物质文化遗产网", "locator": "https://…",
                   "confidence": "high", "claim_scope": "项目身份", "license_status": "internal_paraphrase_only"},
    }


class TestRouting:
    def test_valid_intent(self, fake_llm):
        fake_llm.intent = "qa"
        out = nodes.route_intent({"messages": [HumanMessage(content="河湟皮影是什么？")]})
        assert out["intent"] == "qa"

    def test_garbage_intent_falls_back_to_chat(self, fake_llm):
        fake_llm.intent = "不知道耶"
        out = nodes.route_intent({"messages": [HumanMessage(content="...")]})
        assert out["intent"] == "chat"


class TestQaNode:
    def test_no_evidence_gives_boundary_answer_without_calling_llm(self, fake_llm, monkeypatch):
        monkeypatch.setattr(knowledge, "search_heritage_knowledge", lambda *a, **k: [])
        out = nodes.qa_node({"messages": [HumanMessage(content="凤冠象征什么？")]})
        assert "不能凭空补全" in out["answer"]
        assert out["citations"] == []
        assert not any("只允许基于" in m[0]["content"] for m in fake_llm.calls)  # LLM never asked

    def test_out_of_scope_hits_are_not_evidence(self, fake_llm, monkeypatch):
        monkeypatch.setattr(
            knowledge, "search_heritage_knowledge",
            lambda *a, **k: [_hit(in_scope=False)],
        )
        out = nodes.qa_node({"messages": [HumanMessage(content="某种冷门寓意？")]})
        assert "不能凭空补全" in out["answer"]

    def test_answer_prompt_contains_evidence_and_anti_fabrication_rules(self, fake_llm, monkeypatch):
        monkeypatch.setattr(knowledge, "search_heritage_knowledge", lambda *a, **k: [_hit()])
        out = nodes.qa_node({"messages": [HumanMessage(content="河湟皮影是什么？")]})
        assert out["answer"] == "带引用的回答【1】"
        assert out["citations"][0]["title"] == "河湟皮影戏：国家级非遗项目条目"
        qa_call = next(m for m in fake_llm.calls if m[0]["role"] == "system")
        assert "不得编造" in qa_call[0]["content"]
        assert "【1】" in qa_call[-1]["content"] and "河湟皮影戏属于" in qa_call[-1]["content"]

    def test_project_questions_also_search_project_facts(self, fake_llm, monkeypatch):
        searched = []

        def spy(query, top_k=5, collection="heritage_facts"):
            searched.append(collection)
            return [_hit()] if collection == config.COLLECTION_PROJECT else []

        monkeypatch.setattr(knowledge, "search_heritage_knowledge", spy)
        nodes.qa_node({"messages": [HumanMessage(content="本项目如何查看生成任务状态？")]})
        assert searched == [config.COLLECTION_HERITAGE, config.COLLECTION_PROJECT]


class TestCreateNode:
    def test_missing_lineart_asks_for_upload_and_keeps_slots(self, fake_llm):
        fake_llm.extract_json = '{"role": "女角", "headwear": "凤冠", "color_palette": "暖黄色"}'
        out = nodes.create_node({"messages": [HumanMessage(content="帮我生成一个戴凤冠的女性头像")]})
        assert "上传" in out["answer"] and "线稿" in out["answer"]
        assert out["slots"]["headwear"] == "凤冠"
        assert "generation" not in out

    def test_full_request_generates_and_reports(
        self, fake_llm, monkeypatch, tiny_png
    ):
        fake_llm.extract_json = '{"role": "女角", "headwear": "凤冠", "color_palette": "暖黄色", "facing": "右侧面"}'
        monkeypatch.setattr(
            puppet_api, "generate_shadow_puppet",
            lambda **kw: {"task_id": "t1", "position": 1},
        )
        monkeypatch.setattr(
            puppet_api, "get_generation_status",
            lambda tid: {"task_id": tid, "status": "done",
                         "result": {"final_url": "http://x/final.png", "candidate_urls": ["u1", "u2", "u3"],
                                    "elapsed_seconds": 18.5}},
        )
        monkeypatch.setattr(config, "PUPPET_POLL_INTERVAL", 0)
        out = nodes.create_node(
            {"messages": [HumanMessage(content="帮我生成戴凤冠、右侧面、暖黄色的女角")],
             "lineart_path": str(tiny_png)}
        )
        assert "生成完成" in out["answer"]
        assert out["generation"]["final_url"] == "http://x/final.png"
        assert "凤冠" in out["generation"]["prompt"]

    def test_api_failure_is_surfaced_not_hidden(self, fake_llm, monkeypatch, tiny_png):
        fake_llm.extract_json = '{"role": "女角", "headwear": "凤冠", "color_palette": "暖黄"}'

        def boom(**kw):
            raise puppet_api.PuppetAPIError("皮影生成服务的模型尚未加载或服务不可用，请稍后再试。")

        monkeypatch.setattr(puppet_api, "generate_shadow_puppet", boom)
        out = nodes.create_node(
            {"messages": [HumanMessage(content="生成一个")], "lineart_path": str(tiny_png)}
        )
        assert "生成没有成功" in out["answer"]
        assert "模型尚未加载" in out["answer"]
        assert out["error"]

    def test_failed_task_is_not_reported_as_success(self, fake_llm, monkeypatch, tiny_png):
        fake_llm.extract_json = '{"role": "女角", "headwear": "凤冠", "color_palette": "暖黄"}'
        monkeypatch.setattr(puppet_api, "generate_shadow_puppet", lambda **kw: {"task_id": "t1", "position": 0})
        monkeypatch.setattr(
            puppet_api, "get_generation_status",
            lambda tid: {"task_id": tid, "status": "failed", "error": "CUDA OOM", "result": None},
        )
        monkeypatch.setattr(config, "PUPPET_POLL_INTERVAL", 0)
        out = nodes.create_node(
            {"messages": [HumanMessage(content="生成")], "lineart_path": str(tiny_png)}
        )
        assert "失败" in out["answer"] and "CUDA OOM" in out["answer"]
        assert "生成完成" not in out["answer"]

    def test_poll_timeout_is_explicit(self, fake_llm, monkeypatch, tiny_png):
        fake_llm.extract_json = '{"role": "女角", "headwear": "凤冠", "color_palette": "暖黄"}'
        monkeypatch.setattr(puppet_api, "generate_shadow_puppet", lambda **kw: {"task_id": "t9", "position": 2})
        monkeypatch.setattr(
            puppet_api, "get_generation_status",
            lambda tid: {"task_id": tid, "status": "running", "result": None},
        )
        monkeypatch.setattr(config, "PUPPET_POLL_INTERVAL", 0)
        monkeypatch.setattr(config, "PUPPET_POLL_TIMEOUT", 0.01)
        out = nodes.create_node(
            {"messages": [HumanMessage(content="生成")], "lineart_path": str(tiny_png)}
        )
        assert "t9" in out["answer"] and "超过" in out["answer"]
        assert "生成完成" not in out["answer"]

    def test_slots_merge_across_turns(self):
        first = CreationSlots(role="女角", headwear="凤冠")
        merged = first.merge(CreationSlots(color_palette="暖黄色"))
        assert merged.role == "女角" and merged.color_palette == "暖黄色"
        # None fields never wipe earlier values
        assert first.merge(CreationSlots()).headwear == "凤冠"
        assert build_prompt(merged) == "河湟皮影风格头像，女角，佩戴凤冠，暖黄色色调"

    def test_unparseable_extraction_keeps_previous_slots(self, fake_llm):
        fake_llm.extract_json = "这不是 JSON"
        prev = CreationSlots(role="女角", headwear="凤冠", color_palette="暖黄").model_dump()
        out = nodes.create_node({"messages": [HumanMessage(content="嗯")], "slots": prev})
        # no lineart anywhere -> ask-upload path, previous slots survive
        assert out["slots"]["headwear"] == "凤冠"


class TestHistoryNode:
    def test_history_listed(self, monkeypatch):
        monkeypatch.setattr(
            puppet_api, "get_generation_history",
            lambda limit=5: {"total": 1, "items": [{"task_id": "t1", "status": "done",
                                                    "prompt_preview": "河湟皮影风格头像…"}]},
        )
        out = nodes.history_node({"messages": [HumanMessage(content="看看历史")]})
        assert "t1" in out["answer"]

    def test_history_service_down(self, monkeypatch):
        def boom(limit=5):
            raise puppet_api.PuppetAPIError("无法连接皮影生成服务")

        monkeypatch.setattr(puppet_api, "get_generation_history", boom)
        out = nodes.history_node({"messages": [HumanMessage(content="看看历史")]})
        assert "查不了" in out["answer"]


def _mock_done_service(monkeypatch, captured=None):
    """Mock a working generation service; optionally capture generate kwargs."""
    def fake_generate(**kw):
        if captured is not None:
            captured.update(kw)
        return {"task_id": "t1", "position": 1}

    monkeypatch.setattr(puppet_api, "generate_shadow_puppet", fake_generate)
    monkeypatch.setattr(
        puppet_api, "get_generation_status",
        lambda tid: {"task_id": tid, "status": "done",
                     "result": {"final_url": "http://x/final.png", "candidate_urls": ["u1"],
                                "elapsed_seconds": 16.0}},
    )
    monkeypatch.setattr(config, "PUPPET_POLL_INTERVAL", 0)


class TestCreateNodeRefinements:
    def test_skip_phrase_bypasses_followup_and_generates(self, fake_llm, monkeypatch, tiny_png):
        # 缺 headwear/color，但用户明确说“直接生成”——不许进入追问死循环
        fake_llm.extract_json = "{}"
        _mock_done_service(monkeypatch)
        prev = CreationSlots(role="女角").model_dump()
        out = nodes.create_node(
            {"messages": [HumanMessage(content="直接生成")],
             "slots": prev, "lineart_path": str(tiny_png)}
        )
        assert "生成完成" in out["answer"]

    def test_regen_with_unchanged_slots_bumps_seed(self, fake_llm, monkeypatch, tiny_png):
        fake_llm.extract_json = "{}"
        captured = {}
        _mock_done_service(monkeypatch, captured)
        prev = CreationSlots(role="女角", headwear="凤冠", color_palette="暖黄").model_dump()
        out = nodes.create_node(
            {"messages": [HumanMessage(content="再生成一版")],
             "slots": prev, "lineart_path": str(tiny_png)}
        )
        assert captured["seed"] == config.DEFAULT_SEED + 1
        assert out["slots"]["params"]["seed"] == config.DEFAULT_SEED + 1

    def test_regen_with_changed_slots_keeps_seed(self, fake_llm, monkeypatch, tiny_png):
        fake_llm.extract_json = '{"color_palette": "暖红色"}'
        captured = {}
        _mock_done_service(monkeypatch, captured)
        prev = CreationSlots(role="女角", headwear="凤冠", color_palette="暖黄").model_dump()
        nodes.create_node(
            {"messages": [HumanMessage(content="换成暖红色，再生成一版")],
             "slots": prev, "lineart_path": str(tiny_png)}
        )
        assert captured["seed"] == config.DEFAULT_SEED

    def test_facing_is_disclosed_as_lineart_decided(self, fake_llm, monkeypatch, tiny_png):
        fake_llm.extract_json = '{"role": "女角", "headwear": "凤冠", "color_palette": "暖黄", "facing": "右侧面"}'
        _mock_done_service(monkeypatch)
        out = nodes.create_node(
            {"messages": [HumanMessage(content="生成")], "lineart_path": str(tiny_png)}
        )
        assert "朝向最终以线稿造型为准" in out["answer"]

    def test_prompt_dedupes_style_notes(self):
        slots = CreationSlots(role="女角", extra_notes="皮影风格")
        prompt = build_prompt(slots)
        assert prompt.count("皮影风格") == 1  # 只出现在前缀里，不重复拼接

    def test_trace_records_pipeline_steps(self, fake_llm, monkeypatch, tiny_png):
        fake_llm.extract_json = '{"role": "女角", "headwear": "凤冠", "color_palette": "暖黄"}'
        _mock_done_service(monkeypatch)
        out = nodes.create_node(
            {"messages": [HumanMessage(content="生成")], "lineart_path": str(tiny_png)}
        )
        steps = [t["step"] for t in out["trace"]]
        assert "创作参数" in steps and "工具调用" in steps and "生成状态" in steps


class TestConversationContext:
    def test_route_intent_sees_history(self, fake_llm):
        fake_llm.intent = "qa"
        nodes.route_intent({"messages": [
            HumanMessage(content="河湟皮影戏主要流传在哪里？"),
            AIMessage(content="主要流传于青海河湟地区【1】。"),
            HumanMessage(content="那它的代表性传承人呢？"),
        ]})
        intent_prompt = fake_llm.calls[-1][-1]["content"]
        assert "对话上下文" in intent_prompt and "河湟皮影" in intent_prompt

    def test_qa_retrieval_query_carries_context(self, fake_llm, monkeypatch):
        queries = []

        def spy(query, top_k=5, collection="heritage_facts"):
            queries.append(query)
            return [_hit()]

        monkeypatch.setattr(knowledge, "search_heritage_knowledge", spy)
        nodes.qa_node({"messages": [
            HumanMessage(content="河湟皮影戏主要流传在哪里？"),
            AIMessage(content="主要流传于青海河湟地区【1】。"),
            HumanMessage(content="那传承人呢？"),
        ]})
        # 双路检索：一路原始追问（防止上下文把检索带偏），一路带上下文（消解指代）
        assert queries[0] == "那传承人呢？"
        assert any("河湟皮影" in q and "那传承人呢？" in q for q in queries[1:])

    def test_regen_after_success_is_not_interrogated(self, fake_llm, monkeypatch, tiny_png):
        # 上一轮“直接生成”已成功（缺 color），本轮“再生成一版”不应再被追问拦截
        fake_llm.extract_json = "{}"
        captured = {}
        _mock_done_service(monkeypatch, captured)
        prev = CreationSlots(role="女角", headwear="凤冠", generated_once=True).model_dump()
        out = nodes.create_node(
            {"messages": [HumanMessage(content="再生成一版")],
             "slots": prev, "lineart_path": str(tiny_png)}
        )
        assert "生成完成" in out["answer"]
        assert captured["seed"] == config.DEFAULT_SEED + 1

    def test_success_marks_generated_once(self, fake_llm, monkeypatch, tiny_png):
        fake_llm.extract_json = '{"role": "女角", "headwear": "凤冠", "color_palette": "暖黄"}'
        _mock_done_service(monkeypatch)
        out = nodes.create_node(
            {"messages": [HumanMessage(content="生成")], "lineart_path": str(tiny_png)}
        )
        assert out["slots"]["generated_once"] is True


# ---------------------------------------------------------------------------
# 引用标号校验 + 结构化槽位抽取 + 流式 sink（可靠性加固，2026-08-23）
# ---------------------------------------------------------------------------

class TestCitationSanitize:
    def test_hallucinated_marker_removed(self, fake_llm, monkeypatch):
        # 只有 2 条证据，LLM 却引用了【5】——幻觉标号必须被剔除并留痕
        monkeypatch.setattr(
            knowledge, "search_heritage_knowledge",
            lambda *a, **k: [_hit(doc_id="a"), _hit(doc_id="b", title="另一篇报道")],
        )
        fake_llm.qa_answer = "河湟皮影流传于青海河湟地区【1】。它的纹样非常精美【5】。"
        out = nodes.qa_node({"messages": [HumanMessage(content="河湟皮影是什么？")]})
        assert "【1】" in out["answer"]
        assert "【5】" not in out["answer"]
        assert "纹样非常精美。" in out["answer"]  # 文本本体保留
        assert any(t["step"] == "引用校验" and "5" in t["detail"] for t in out["trace"])

    def test_valid_markers_untouched(self, fake_llm, monkeypatch):
        monkeypatch.setattr(knowledge, "search_heritage_knowledge", lambda *a, **k: [_hit()])
        fake_llm.qa_answer = "河湟皮影流传于青海【1】。"
        out = nodes.qa_node({"messages": [HumanMessage(content="河湟皮影流传在哪？")]})
        assert out["answer"] == "河湟皮影流传于青海【1】。"
        assert not any(t["step"] == "引用校验" for t in out["trace"])


class TestSlotExtractionRobust:
    def test_json_mode_and_retry_on_garbage(self, monkeypatch):
        calls = []
        outputs = ["抱歉我没听懂你在说什么", '{"role": "女角", "headwear": "凤冠"}']

        def fake_chat(messages, **kwargs):
            calls.append(kwargs.get("json_mode"))
            return outputs[len(calls) - 1]

        monkeypatch.setattr(nodes, "llm_chat", fake_chat)
        slots = nodes._extract_slots("我想要一个戴凤冠的女角")
        assert slots.role == "女角" and slots.headwear == "凤冠"
        assert len(calls) == 2                    # 第一次垃圾输出触发了纠偏重试
        assert all(c is True for c in calls)      # 两次都走 json_mode

    def test_validation_error_retries_then_empty(self, monkeypatch):
        # 类型错误（role 是 list）两次都失败 → 回退空槽位，不抛异常
        monkeypatch.setattr(nodes, "llm_chat", lambda *a, **k: '{"role": ["女角"]}')
        slots = nodes._extract_slots("我想要一个女角")
        assert isinstance(slots, CreationSlots)
        assert slots.role is None


class TestQaStreaming:
    def test_qa_streams_tokens_to_sink(self, fake_llm, monkeypatch):
        monkeypatch.setattr(knowledge, "search_heritage_knowledge", lambda *a, **k: [_hit()])
        monkeypatch.setattr(nodes, "llm_chat_stream",
                            lambda messages: iter(["河湟皮影", "流传于青海", "【1】"]))
        sink: list[str] = []
        nodes.set_stream_sink(sink)
        try:
            out = nodes.qa_node({"messages": [HumanMessage(content="河湟皮影是什么？")]})
        finally:
            nodes.clear_stream_sink()
        assert sink == ["河湟皮影", "流传于青海", "【1】"]   # 逐 token 到达
        assert out["answer"] == "河湟皮影流传于青海【1】"  # 状态里仍是完整回答
        assert out["citations"]
