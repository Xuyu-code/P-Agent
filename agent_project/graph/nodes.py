"""Graph nodes: intent routing, evidence-bounded QA, creation slot filling.

LLM access goes through llm.chat (one monkeypatch point for tests); tool
access goes through the tools package modules (another). Nodes never invent
cultural facts — the QA prompt below encodes the evidence-bounded policy
from data_sources/SOURCE_CARDS_README.md.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from pydantic import ValidationError

from .. import config
from ..llm import chat as llm_chat
from ..llm import chat_stream as llm_chat_stream
from ..schemas.creation import CreationSlots, build_prompt
from ..tools import knowledge, puppet_api
from .state import AgentState

# ---------------------------------------------------------------------------
# 流式输出通道：调用方（Streamlit / FastAPI SSE）在 invoke 前挂上 sink
# （list 或 queue），qa_node 生成时逐 token 写入；None = 普通整段返回。
# ---------------------------------------------------------------------------

_stream_sink = None


def set_stream_sink(sink) -> None:
    global _stream_sink
    _stream_sink = sink


def clear_stream_sink() -> None:
    global _stream_sink
    _stream_sink = None


def _sink_put(tok: str) -> None:
    put = getattr(_stream_sink, "put", None) or getattr(_stream_sink, "append")
    put(tok)

# ---------------------------------------------------------------------------
# Intent routing
# ---------------------------------------------------------------------------

_INTENT_PROMPT = """你是意图分类器。把用户的最新一句话分类为以下四类之一，只输出标签本身：

- qa：询问河湟皮影/非物质文化遗产/相关人物（如传承人"某某是谁"）或本项目使用方式的知识性问题
- create：想生成、绘制、修改一个皮影头像（包括"帮我生成…""再改一版…"）；
  若用户正在进行创作流程，其补充说明（如"线稿传好了""直接生成""换成红色"）也算 create
- history：想查看之前生成过的作品、历史记录或任务状态（如"最近有哪些任务""上次那张好了吗"）
- chat：打招呼、闲聊或无法归入以上三类
{context}
用户：{message}
标签："""

_VALID_INTENTS = {"qa", "create", "history", "chat"}


def _recent_context(state: AgentState, turns: int = 2) -> str:
    """Last N user/assistant exchanges (excluding the current message).

    Lets follow-up questions ("那它的传承人呢？") retrieve and route sensibly.
    """
    msgs = state.get("messages", [])[:-1]
    lines = []
    for m in msgs[-2 * turns:]:
        role = "用户" if m.type == "human" else "助手"
        lines.append(f"{role}：{m.content[:120]}")
    return "\n".join(lines)


def route_intent(state: AgentState) -> dict[str, Any]:
    message = state["messages"][-1].content
    # 创作流程进行中的上下文，帮助跟进式短句（"线稿传好了"）正确分流
    context = ""
    history = _recent_context(state)
    if history:
        context += f"\n（对话上下文：\n{history}\n）"
    if state.get("slots"):
        context += "\n（背景：用户正在创作流程中，已记录部分创作参数。）"
    if state.get("lineart_path"):
        context += "\n（背景：用户已上传线稿。）"
    label = llm_chat([{"role": "user", "content": _INTENT_PROMPT.format(message=message, context=context)}])
    label = label.strip().strip("`。.").lower()
    intent = label if label in _VALID_INTENTS else "chat"
    return {"intent": intent, "trace": [{"step": "意图识别", "detail": intent}]}


def intent_edge(state: AgentState) -> str:
    return state.get("intent", "chat")


# ---------------------------------------------------------------------------
# QA node (evidence-bounded)
# ---------------------------------------------------------------------------

_QA_SYSTEM = """你是"河湟非遗知识问答"助手，只能基于给定证据回答：
1. 证据没有支持的寓意、数字、年代一律不得编造；证据不足就明说"现有公开资料摘要中没有支持这一点"，并建议补充传承人口述或授权文献，不要硬答。
2. 单篇报道的细节用"据《标题》报道介绍"限定，不得说成所有流派统一适用。
3. project_document 证据表述为"本项目公开说明显示"，不得说成传统史料或民间记载。
4. 引用事实在句末标注【N】（N 为证据编号），末尾列"参考来源"（标题＋发布机构）。"""

_QA_USER = """{context}用户问题：{question}

证据：
{evidence}

请回答（遵守系统规则；若问题中有指代（如"它"、"他"），结合对话上下文理解）："""

_NO_EVIDENCE_ANSWER = (
    "抱歉，现有资料无法回答这个问题。我的知识库只收录了有限公开资料的自写摘要，"
    "其中没有支持这一点的证据，我不能凭空补全。\n\n"
    "如果你能提供传承人说明、剧目资料、馆藏说明或授权文献，我可以整理后纳入知识库再回答。"
)

# 引用标号校验：LLM 写的【N】必须在实际证据数量之内，否则是幻觉标号，剔除并留痕
_CITE_RE = re.compile(r"【(\d+)】")


def _sanitize_citation_markers(answer: str, n_evidence: int) -> tuple[str, list[int]]:
    invalid: list[int] = []

    def repl(m: re.Match) -> str:
        n = int(m.group(1))
        if 1 <= n <= n_evidence:
            return m.group(0)
        invalid.append(n)
        return ""

    return _CITE_RE.sub(repl, answer), invalid

# Questions about the public service itself also search the project collection.
_PROJECT_HINT_RE = re.compile(r"接口|API|本项目|系统|生成|候选|服务|使用|状态", re.I)

# The service collection may contain English endpoint terminology. Add a small
# bilingual gloss for stable retrieval across Chinese and English questions.
_BILINGUAL_TERMS = {
    "参数": "parameters", "接口": "API", "服务": "service",
    "候选": "candidate", "状态": "status",
}


def _augment_query(question: str) -> str:
    glosses = sorted({en for zh, en in _BILINGUAL_TERMS.items() if zh in question})
    return f"{question} {' '.join(glosses)}" if glosses else question


def _search_merged(question: str, context: str, collection: str, top_k: int, *, augment: bool = False) -> list[dict]:
    """Dual-query retrieval with per-query quota.

    Follow-ups with anaphora ("那它的传承人呢？") need context to be
    interpretable, but a context-heavy query drifts toward the previous topic.
    A global top-k after merging still lets the context query crowd out the
    raw question's hits (实测：传承人卡片 0.63 被上下文查询的地区卡片挤出前 5）。
    So each query gets its own guaranteed quota, then the union is sorted.
    """
    queries = [question] + ([f"{context}\n{question}"] if context else [])
    # 双路时才拆分配额；单路（首轮无上下文）必须取满 top_k——否则 top-5 的
    # 目标卡会被 quota=3 切掉（实测：02 卡排第 4 被裁，LLM 只能保守拒答）
    quota = max(3, top_k // 2) if len(queries) > 1 else top_k
    best: dict[str, dict] = {}
    for q in queries:
        if augment:
            q = _augment_query(q)
        for h in knowledge.search_heritage_knowledge(q, top_k=quota, collection=collection):
            cur = best.get(h["doc_id"])
            if cur is None or h["score"] > cur["score"]:
                best[h["doc_id"]] = h
    return sorted(best.values(), key=lambda h: h["score"], reverse=True)[:top_k + (2 if context else 0)]


def _format_evidence(hits: list[dict], offset: int = 0) -> str:
    lines = []
    for i, hit in enumerate(hits, start=offset + 1):
        src = hit["source"]
        lines.append(
            f"【{i}】（{src.get('title')}／{src.get('publisher')}）\n{hit['snippet']}"
        )
    return "\n\n".join(lines)


def qa_node(state: AgentState) -> dict[str, Any]:
    question = state["messages"][-1].content
    context = _recent_context(state)

    heritage_hits = _search_merged(question, context, config.COLLECTION_HERITAGE, config.RETRIEVAL_TOP_K)
    hits = [h for h in heritage_hits if h["in_scope"]]
    project_hits: list[dict] = []
    if _PROJECT_HINT_RE.search(question):
        # Service documents use the same bounded retrieval path as other
        # project-owned references.
        project_hits = _search_merged(question, context, config.COLLECTION_PROJECT, 5, augment=True)
        hits.extend(h for h in project_hits if h["in_scope"])

    sources = sorted({h["source"].get("publisher", "?") for h in hits})
    trace = list(state.get("trace", []))
    trace.append({
        "step": "知识检索",
        "detail": f"heritage_facts 命中 {sum(1 for h in heritage_hits if h['in_scope'])} 条"
                  + (f"，project_facts 命中 {sum(1 for h in project_hits if h['in_scope'])} 条" if project_hits else "")
                  + f"；来源：{'、'.join(sources) if sources else '无（未越过证据阈值）'}",
    })

    if not hits:
        return {"answer": _NO_EVIDENCE_ANSWER, "evidence": [], "citations": [], "trace": trace}

    context_block = f"对话上下文：\n{context}\n\n" if context else ""
    prompt = _QA_USER.format(question=question, evidence=_format_evidence(hits), context=context_block)
    qa_messages = [{"role": "system", "content": _QA_SYSTEM}, {"role": "user", "content": prompt}]
    if _stream_sink is not None:
        chunks: list[str] = []
        for tok in llm_chat_stream(qa_messages):
            chunks.append(tok)
            _sink_put(tok)
        answer = "".join(chunks)
    else:
        answer = llm_chat(qa_messages)
    answer, invalid_cites = _sanitize_citation_markers(answer, len(hits))
    if invalid_cites:
        trace.append({
            "step": "引用校验",
            "detail": f"剔除幻觉引用标号 {invalid_cites}（超出证据数量 {len(hits)}）",
        })
    citations = [
        {"n": i, "title": h["source"].get("title"), "publisher": h["source"].get("publisher"),
         "locator": h["source"].get("locator"), "confidence": h["source"].get("confidence")}
        for i, h in enumerate(hits, start=1)
    ]
    return {"answer": answer, "evidence": hits, "citations": citations, "trace": trace}


# ---------------------------------------------------------------------------
# Creation node
# ---------------------------------------------------------------------------

_EXTRACT_PROMPT = """从对话中提取用户想要的皮影头像信息，输出 JSON（没有的字段写 null，不要臆造）：

{{
  "role": "角色，如 女角/老生/小生",
  "facing": "朝向，如 左侧面/右侧面",
  "headwear": "冠饰，如 凤冠",
  "color_palette": "色彩，如 暖黄色",
  "pattern": "纹样，如 镂空云纹",
  "extra_notes": "其他描述",
  "negative_prompt": "用户明确要求不要出现的内容"
}}

只输出 JSON。用户最新一句话：{message}"""


def _parse_slots_json(raw: str) -> CreationSlots | None:
    """json_mode 下应为纯 JSON；解析或 Pydantic 校验失败返回 None（不静默吞错，
    由调用方决定是否带纠偏提示重试）。"""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)  # 兜底：少数输出仍带闲聊包装
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    try:
        return CreationSlots.model_validate(
            {k: v for k, v in data.items() if k in CreationSlots.model_fields}
        )
    except ValidationError:
        return None


def _extract_slots(message: str) -> CreationSlots:
    prompt = _EXTRACT_PROMPT.format(message=message)
    slots = _parse_slots_json(llm_chat([{"role": "user", "content": prompt}], json_mode=True))
    if slots is None:
        # 带纠偏提示重试一次；再失败才回退空槽位
        retry_prompt = (prompt + "\n\n上次输出无法解析为合法 JSON。"
                        "请只输出符合结构的 JSON 对象，不要输出任何其他内容。")
        slots = _parse_slots_json(llm_chat([{"role": "user", "content": retry_prompt}], json_mode=True))
    return slots or CreationSlots()


def _describe_defaults(slots: CreationSlots) -> str:
    """Disclose which fields were defaulted, so nothing is silently invented."""
    notes = []
    if not slots.facing:
        notes.append("朝向未指定（由线稿决定）")
    else:
        # 生成服务以条件图为准：prompt 里的朝向拗不过线稿造型，如实说明
        notes.append("朝向最终以线稿造型为准")
    if not slots.pattern:
        notes.append("纹样未指定（由服务根据输入补全）")
    return "；".join(notes)


# 用户明确跳过补充说明（“直接生成”）——不进入追问死循环
_SKIP_RE = re.compile(r"直接生成|就这样|开始生成|可以了|不用(?:再?问|补充)|按(?:默认|现有)")

# 用户要求“再来一版”且槽位没变时，seed +1，否则同一 seed 会产出同一张图
_REGEN_RE = re.compile(r"再生成|再来一|重新生成|再改一版|换一版|换一批|再试一|再出一")

_DESCRIPTIVE_FIELDS = ("role", "facing", "headwear", "color_palette", "pattern", "extra_notes", "negative_prompt")


def create_node(state: AgentState) -> dict[str, Any]:
    message = state["messages"][-1].content

    previous = CreationSlots(**state["slots"]) if state.get("slots") else CreationSlots()
    delta = _extract_slots(message)
    slots = previous.merge(delta)
    if state.get("lineart_path"):
        slots.lineart_path = state["lineart_path"]

    trace = list(state.get("trace", []))
    filled = {k: getattr(slots, k) for k in _DESCRIPTIVE_FIELDS if getattr(slots, k)}
    trace.append({
        "step": "创作参数",
        "detail": "；".join(f"{k}={v}" for k, v in filled.items()) or "（尚未提取到描述字段）",
    })

    out: dict[str, Any] = {"slots": slots.model_dump(), "trace": trace}

    # Hard requirement: the generation service is condition-driven.
    if not slots.lineart_path:
        out["answer"] = (
            "好的，我来帮你创作皮影头像。生成是以线稿为条件的，请先上传一张**头像线稿**"
            "（侧面头部的黑白线稿，PNG/JPEG/WEBP/BMP，不超过 5MB）。\n\n"
            "⚠️ 注意：当前服务针对河湟皮影**头像**创作优化；全身像、照片或其他内容的图"
            "可能不适合此工作流，结果质量会受到影响。"
            + (f"\n\n我已记下：{build_prompt(slots)}。" if slots.role or slots.headwear else "")
        )
        return out

    missing = slots.missing_fields()
    # 已用当前槽位成功生成过（用户事实上认可了现有信息）、或本轮明确说“直接
    # 生成/再来一版”时，不再追问缺省字段——否则“再生成一版”会被追问拦截。
    consented = previous.generated_once or bool(_SKIP_RE.search(message)) or bool(_REGEN_RE.search(message))
    if missing and not slots.extra_notes and not consented:
        labels = {"role": "角色（如女角、老生）", "headwear": "冠饰（如凤冠）", "color_palette": "色彩倾向"}
        wanted = "、".join(labels[m] for m in missing)
        out["answer"] = (
            f"线稿已收到。为了让生成更准确，请再补充：{wanted}。\n"
            "如果暂时不想细说，回复“直接生成”，我就按当前信息并配合默认参数开始。"
        )
        return out

    # “再生成一版”且描述字段没有任何变化：换 seed，否则结果与上一版完全相同
    seed_note = ""
    if _REGEN_RE.search(message) and state.get("slots"):
        prev_descriptive = {k: getattr(previous, k) for k in _DESCRIPTIVE_FIELDS}
        new_descriptive = {k: getattr(slots, k) for k in _DESCRIPTIVE_FIELDS}
        if prev_descriptive == new_descriptive:
            slots.params.seed = previous.params.seed + 1
            out["slots"] = slots.model_dump()
            seed_note = f"（已换新随机种子 seed={slots.params.seed}）"
            trace.append({"step": "创作参数", "detail": f"描述未变，seed {previous.params.seed} → {slots.params.seed}"})

    prompt = build_prompt(slots)
    trace.append({"step": "工具调用", "detail": f"POST {config.PUPPET_API_BASE}/api/generate（prompt：{prompt}）"})
    try:
        submitted = puppet_api.generate_shadow_puppet(
            prompt=prompt,
            lineart_path=slots.lineart_path,
            negative_prompt=slots.negative_prompt,
            **slots.params.model_dump(),
        )
        trace.append({"step": "工具调用", "detail": f"任务已受理：task_id={submitted['task_id']}（队列第 {submitted.get('position', '?')} 位）"})
        result = _wait_for_task(submitted["task_id"])
        trace.append({"step": "生成状态", "detail": result["status"]})
    except puppet_api.PuppetAPIError as exc:
        trace.append({"step": "生成状态", "detail": "failed（提交阶段错误）"})
        out["answer"] = f"生成没有成功：{exc}"
        out["error"] = str(exc)
        return out

    out["generation"] = {"prompt": prompt, **result}
    if result["status"] == "done" and result.get("result"):
        slots.generated_once = True
        out["slots"] = slots.model_dump()
        final_url = result["result"].get("final_url")
        defaults_note = _describe_defaults(slots)
        out["answer"] = (
            f"生成完成 ✅\n\n提示词：{prompt}{seed_note}\n"
            f"耗时约 {result['result'].get('elapsed_seconds', '?')} 秒，"
            f"系统返回 {len(result['result'].get('candidate_urls', []))} 张候选并标记推荐结果。"
            + (f"\n（{defaults_note}）" if defaults_note else "")
            + "\n\n想调整的话直接告诉我，比如“换成暖红色”、“再生成一版”。"
        )
        out["generation"]["final_url"] = final_url
    elif result["status"] == "failed":
        out["answer"] = f"生成任务失败了：{result.get('error') or '服务未返回具体原因'}。可以让我重试。"
        out["error"] = result.get("error")
    else:  # timed out while polling — the task may still finish server-side
        out["answer"] = (
            f"任务已提交（task_id: {result['task_id']}），但等待超过了本地时限。"
            "生成服务串行排队，任务可能仍在后台进行，稍后问我“上次的结果出来了吗”即可。"
        )
    return out


def _wait_for_task(task_id: str) -> dict[str, Any]:
    """Poll until done/failed or PUPPET_POLL_TIMEOUT; never fabricates success."""
    deadline = time.monotonic() + config.PUPPET_POLL_TIMEOUT
    while time.monotonic() < deadline:
        payload = puppet_api.get_generation_status(task_id)
        if payload["status"] in ("done", "failed"):
            return payload
        time.sleep(config.PUPPET_POLL_INTERVAL)
    return {"task_id": task_id, "status": "timeout", "result": None}


# ---------------------------------------------------------------------------
# History node
# ---------------------------------------------------------------------------

def history_node(state: AgentState) -> dict[str, Any]:
    trace = list(state.get("trace", []))
    trace.append({"step": "工具调用", "detail": f"GET {config.PUPPET_API_BASE}/api/history"})
    try:
        payload = puppet_api.get_generation_history(limit=5)
    except puppet_api.PuppetAPIError as exc:
        return {"answer": f"暂时查不了历史记录：{exc}", "error": str(exc), "trace": trace}
    items = payload.get("items", [])
    trace.append({"step": "查询结果", "detail": f"返回 {len(items)} 条（共 {payload.get('total', '?')} 条）"})
    if not items:
        return {"answer": "皮影生成服务里还没有历史任务。", "generation": {"history": []}, "trace": trace}
    lines = []
    for item in items:
        status_cn = {"done": "✅", "failed": "❌", "running": "⏳", "queued": "🕐"}.get(item["status"], item["status"])
        lines.append(f"- {status_cn} `{item['task_id']}`：{item.get('prompt_preview', '')}")
    return {
        "answer": "最近的生成任务：\n" + "\n".join(lines),
        "generation": {"history": items},
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Chat node
# ---------------------------------------------------------------------------

def chat_node(state: AgentState) -> dict[str, Any]:
    return {
        "answer": (
            "你好！我是河湟非遗知识问答与皮影创作助手。我可以：\n"
            "1. **答**：基于公开资料摘要回答河湟皮影的项目背景、造型结构、传承等问题，并给出引用来源；\n"
            "2. **创**：根据你的描述（角色、朝向、冠饰、色彩等）调用皮影生成服务创作头像——需要先上传一张线稿；\n"
            "3. **查**：查看最近的生成历史。\n\n"
            "想从哪里开始？"
        )
    }
