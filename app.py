"""Streamlit UI: a conversational heritage consultant, not an upload tool.

设计原则（与外部图像生成服务解耦）：
- 对话是主角：用户先进来聊天/提问，创作在多轮对话中自然展开；
- 上传线稿不是入口，只是 Agent 在创作条件成熟时才要求的一个工具输入；
- 每一轮的"处理过程"（意图识别 → 知识检索 → 工具调用 → 生成状态）可见，
  体现这是一个会问答、会检索、会规划、会调用的 Agent，而不是生成器外壳。

Run from the project root:
    streamlit run app.py --server.headless true     # default port 8501
"""

from __future__ import annotations

import base64
import json
import os
import queue
import threading
import time
import uuid
from pathlib import Path

import httpx
import streamlit as st
import streamlit.components.v1 as components
from langchain_core.messages import AIMessage, HumanMessage

from agent_project import config
from agent_project.graph import nodes
from agent_project.graph.builder import get_graph
from agent_project.tools import puppet_api

st.set_page_config(
    page_title="河湟非遗皮影智能创作 Agent",
    page_icon=str(Path(__file__).parent / "assets" / "seal_logo.png"),  # 浏览器标签页用“影”字印章
    layout="wide",
)

# ---------------------------------------------------------------------------
# 皮影风格界面皮肤（仅外观，不影响任何逻辑）
# 主题色另见 .streamlit/config.toml（重启后生效），此处 CSS 立即生效。
# ---------------------------------------------------------------------------


def inject_puppet_style() -> None:
    """注入皮影风格 CSS：宣纸底 + 朱红 + 泥金 + 黛蓝 + 墨色。"""
    css_file = Path(__file__).parent / "assets" / "puppet_style.css"
    if css_file.exists():
        st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def inject_chinese_toolbar_labels() -> None:
    """把 Streamlit 内置菜单（Rerun/Clear cache/Deploy 等）实时替换为中文。

    官方没有菜单汉化配置项（文案硬编码在前端 bundle）。这里利用组件 iframe
    与主页面同源这一点，在父页面挂 MutationObserver 做文案替换——非官方手段，
    Streamlit 升级后若 DOM 结构变化导致失效，删除本函数调用即可，不影响功能。
    """
    components.html(
        """
        <script>
        (function () {
          const doc = window.parent && window.parent.document;
          if (!doc || window.parent.__zhToolbarInstalled) return;
          window.parent.__zhToolbarInstalled = true;
          const LABELS = {
            "Rerun": "重新运行",
            "Auto rerun": "自动重跑",
            "Clear cache": "清除缓存",
            "Print": "打印",
            "Record screen": "录屏",
            "Deploy": "部署",
            "Stop": "停止",
            "Settings": "设置",
            "Keyboard shortcuts": "键盘快捷键",
            "About": "关于"
          };
          function translate(node) {
            if (node.nodeType === Node.TEXT_NODE) {
              const t = node.textContent.trim();
              if (LABELS[t]) {
                node.textContent = node.textContent.replace(t, LABELS[t]);
              } else if (t.startsWith("Made with Streamlit")) {
                node.textContent = node.textContent.replace("Made with Streamlit", "由 Streamlit 驱动");
              }
            } else if (node.nodeType === Node.ELEMENT_NODE) {
              node.childNodes.forEach(translate);
            }
          }
          function boot() {
            translate(doc.body);
            new MutationObserver((ms) =>
              ms.forEach((m) => m.addedNodes.forEach(translate))
            ).observe(doc.body, { childList: true, subtree: true });
          }
          if (doc.body) boot();
          else doc.addEventListener("DOMContentLoaded", boot);
        })();
        </script>
        """,
        height=0,
    )


def _asset_b64(name: str) -> str:
    """base64 内联 assets/ 里的图片（皮影人物、印章），供 HTML 块使用。"""
    p = Path(__file__).parent / "assets" / name
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


# 回纹（方形万字不到头）SVG 纹样带：base64 内联，避免 data-URI 引号转义地狱
_MEANDER_URI = "data:image/svg+xml;base64," + base64.b64encode(
    "<svg xmlns='http://www.w3.org/2000/svg' width='20' height='10'>"
    "<path d='M0 9 H18 V1 H4 V7 H12 V3 H8 V5' stroke='#C9A227' fill='none'/></svg>".encode()
).decode()


def render_page_header() -> None:
    """页面顶部：印章 + 标题 + 右侧皮影人物（puppet_header.webp，
    与欢迎页主视觉 puppet_figure.webp 区分，避免同图重复）；
    泥金双划线下叠一条回纹带，戏台台口的感觉。"""
    seal_b64 = _asset_b64("seal_logo.png")
    if seal_b64:
        seal_html = (
            f"<img src='data:image/png;base64,{seal_b64}' "
            "style='width:72px;height:72px;border-radius:10px'/>"
        )
    else:
        seal_html = (
            "<div style='width:64px;height:64px;background:#B5342A;color:#FFFBF2;"
            "font-size:34px;display:flex;align-items:center;justify-content:center;"
            "border-radius:10px;font-family:KaiTi,STKaiti,serif'>影</div>"
        )
    figure_b64 = _asset_b64("puppet_header.webp")
    figure_html = (
        f"<img src='data:image/webp;base64,{figure_b64}' "
        "style='height:96px;margin-left:auto;opacity:0.95'/>"
        if figure_b64 else ""
    )
    st.markdown(
        "<div style='display:flex;align-items:flex-end;gap:18px'>"
        + seal_html
        + "<h1 style='margin:0;border:none;padding:0 0 6px'>河湟非遗皮影智能创作 Agent</h1>"
        + figure_html
        + "</div>"
        + "<div style='border-bottom:3px double #C9A227'></div>"
        + f"<div style='height:10px;margin:3px 0 10px;opacity:0.55;"
          f"background-image:url(\"{_MEANDER_URI}\");background-repeat:repeat-x'></div>",
        unsafe_allow_html=True,
    )


inject_puppet_style()
inject_chinese_toolbar_labels()

config.ensure_data_dirs()

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def _init_state() -> None:
    st.session_state.setdefault("chat_log", [])       # {role, content, citations, generation, trace, elapsed}
    st.session_state.setdefault("slots", None)        # serialized CreationSlots carried across turns
    st.session_state.setdefault("lineart_path", None) # uploaded lineart for this session
    st.session_state.setdefault("pending_input", None)
    st.session_state.setdefault("api_session_id", None)


_init_state()

# ---------------------------------------------------------------------------
# API 客户端模式（dogfooding）：8001 在线时，本界面只是它的前端——
# 对话走 HTTP/SSE，会话落库在服务端，刷新页面经 URL 参数 ?s= 续接记录。
# 8001 不在线时自动退回进程内直连（功能一致，但记录不持久）。
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("AGENT_API_BASE", "http://localhost:8001").rstrip("/")


def _api_alive() -> bool:
    try:
        return httpx.get(f"{API_BASE}/api/health", timeout=1.5).status_code == 200
    except Exception:
        return False


def _api_init() -> None:
    """检测 API 模式；在线则经 URL 参数恢复会话记录（刷新不丢）。"""
    if st.session_state.get("_api_ready"):
        return
    st.session_state._api_ready = True
    st.session_state.api_mode = _api_alive()
    if not st.session_state.api_mode:
        return
    sid = st.query_params.get("s")
    restored = False
    if sid:
        try:
            resp = httpx.get(f"{API_BASE}/api/sessions/{sid}/messages", timeout=5)
            if resp.status_code == 200:
                st.session_state.api_session_id = sid
                st.session_state.chat_log = [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "citations": m.get("citations") or [],
                        "generation": m.get("generation") or {},
                        "trace": m.get("trace") or [],
                        "elapsed": m.get("latency_s"),
                    }
                    for m in resp.json()["messages"]
                ]
                sess = httpx.get(f"{API_BASE}/api/sessions/{sid}", timeout=5).json()
                st.session_state.slots = sess.get("slots") or None
                st.session_state.lineart_path = sess.get("lineart_path")
                restored = True
        except Exception:
            restored = False
    if not restored:
        # 开新会话并写进 URL，刷新后可找回
        try:
            new_sid = httpx.post(f"{API_BASE}/api/sessions", timeout=5).json()["session_id"]
            st.session_state.api_session_id = new_sid
            st.query_params["s"] = new_sid
        except Exception:
            st.session_state.api_mode = False


def _api_stream_turn(sid: str, message: str):
    """经 8001 SSE 完成一轮对话。yield ("token", str) / ("done", dict)。"""
    with httpx.stream(
        "POST", f"{API_BASE}/api/chat/stream",
        json={"message": message, "session_id": sid},
        timeout=httpx.Timeout(300.0, connect=10.0),
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            evt = json.loads(line[6:])
            if evt["type"] == "token":
                yield ("token", evt["content"])
            elif evt["type"] == "done":
                yield ("done", evt)
            elif evt["type"] == "error":
                raise RuntimeError(evt.get("message", "服务错误"))


_api_init()

_SLOT_LABELS = {
    "role": "角色", "facing": "朝向", "headwear": "冠饰",
    "color_palette": "色彩", "pattern": "纹样", "extra_notes": "其他",
}


# ---------------------------------------------------------------------------
# Renderers (shared by chat history replay and the live turn)
# ---------------------------------------------------------------------------


def render_trace(trace: list[dict]) -> None:
    if not trace:
        return
    with st.expander("🛠 本轮处理过程", expanded=False):
        for t in trace:
            st.markdown(f"- **{t['step']}**：{t['detail']}")


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander("📚 参考来源"):
        for c in citations:
            st.caption(
                f"【{c['n']}】{c.get('title')}／{c.get('publisher')}"
                + (f" — {c.get('locator')}" if c.get("locator") else "")
            )


def render_generation(gen: dict) -> None:
    if gen.get("final_url"):
        st.image(gen["final_url"], width=320, caption="成品（系统推荐）")
    result = gen.get("result") or {}
    candidates = result.get("candidate_urls") or []
    if candidates:
        selected = result.get("selected_index", 0)
        with st.expander(f"🔍 候选对比（{len(candidates)} 张，选中第 {selected + 1} 张）"):
            cols = st.columns(len(candidates))
            for i, url in enumerate(candidates):
                with cols[i]:
                    st.image(url, use_container_width=True)
                    mark = "⭐ " if i == selected else ""
                    st.caption(f"{mark}候选 {i + 1}")
    for item in gen.get("history") or []:
        pass  # history entries already rendered inside the answer text


def render_entry(entry: dict) -> None:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        render_generation(entry.get("generation") or {})
        render_citations(entry.get("citations") or [])
        render_trace(entry.get("trace") or [])
        if entry.get("elapsed") is not None:
            st.caption(f"本轮耗时 {entry['elapsed']:.1f}s")


def _local_turn(state_in: dict):
    """进程内直连模式（8001 不在线时的后备）：线程跑图 + sink 流式渲染。"""
    started = time.perf_counter()
    token_q: queue.Queue = queue.Queue()
    holder: dict = {}

    def _work():
        try:
            nodes.set_stream_sink(token_q)
            holder["result"] = get_graph().invoke(state_in)
        finally:
            nodes.clear_stream_sink()
            token_q.put(None)

    th = threading.Thread(target=_work)
    th.start()
    with st.spinner("思考中…（创作请求需等待生成，约 15–25 秒）"):
        first_tok = token_q.get()
    if first_tok is None:
        result = holder["result"]
        st.markdown(result.get("answer") or "（没有生成回答）")
    else:
        def _stream_gen():
            yield first_tok
            while True:
                t = token_q.get()
                if t is None:
                    break
                yield t
        st.write_stream(_stream_gen())
        result = holder["result"]
    th.join()
    return result, time.perf_counter() - started


def _api_turn(user_input: str):
    """API 客户端模式：经 8001 SSE 完成一轮（服务端落库，刷新不丢记录）。"""
    started = time.perf_counter()
    stream = _api_stream_turn(st.session_state.api_session_id, user_input)
    with st.spinner("思考中…（创作请求需等待生成，约 15–25 秒）"):
        first_kind, first_payload = next(stream)
    done = None
    if first_kind == "done":
        done = first_payload
        st.markdown(done["answer"])
    else:
        def _gen():
            nonlocal done
            yield first_payload
            for kind, payload in stream:
                if kind == "token":
                    yield payload
                else:
                    done = payload
        st.write_stream(_gen())
    result = {
        "answer": done["answer"],
        "citations": done["citations"],
        "generation": done["generation"],
        "trace": done["trace"],
        "slots": done.get("slots"),
    }
    return result, done.get("latency_s") or (time.perf_counter() - started)


# ---------------------------------------------------------------------------
# Sidebar: status panels (secondary; never the protagonist)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ 运行状态")
    if st.button("刷新服务状态"):
        st.session_state.pop("health", None)
    if "health" not in st.session_state:
        try:
            st.session_state["health"] = puppet_api.check_health()
        except puppet_api.PuppetAPIError as exc:
            st.session_state["health"] = {"error": str(exc)}
    health = st.session_state["health"]
    if "error" in health:
        st.error(health["error"])
    else:
        ok = health.get("model_loaded")
        st.write(f"生成服务：{'✅ 模型已加载' if ok else '⚠️ 模型未加载'}"
                 + (f"（{health.get('gpu_name')}）" if health.get("gpu_name") else ""))
        st.write(f"队列：{health.get('queue_length', 0)} 个任务")
    st.caption(f"生成服务：{config.PUPPET_API_BASE} · LLM：{config.LLM_MODEL} · 嵌入：{config.EMBEDDING_PROVIDER}")
    st.caption(
        "服务模式：API 客户端（会话落库，刷新不丢）" if st.session_state.get("api_mode")
        else "服务模式：本地直连（8001 未启动，记录不落库）"
    )

    st.divider()
    st.header("🎨 当前创作")
    slots = st.session_state.slots
    if slots:
        rows = [f"- {label}：{slots.get(key)}" for key, label in _SLOT_LABELS.items() if slots.get(key)]
        st.markdown("\n".join(rows) if rows else "（还没有提取到创作描述）")
        if st.session_state.lineart_path:
            st.caption(f"线稿：{Path(st.session_state.lineart_path).name}")
        if st.button("重置创作参数"):
            st.session_state.slots = None
            st.rerun()
    else:
        st.caption("还没有开始创作。直接在对话里描述你想要的皮影头像即可。")

    if st.button("🧹 清空对话"):
        st.session_state.chat_log = []
        st.session_state.slots = None
        st.session_state.lineart_path = None
        st.session_state.pending_input = None
        if st.session_state.get("api_mode"):
            # 新开会话（旧会话仍留在服务端数据库，可追溯）
            try:
                new_sid = httpx.post(f"{API_BASE}/api/sessions", timeout=5).json()["session_id"]
                st.session_state.api_session_id = new_sid
                st.query_params["s"] = new_sid
            except Exception:
                st.session_state.api_mode = False
        st.rerun()

    with st.expander("知识库"):
        st.caption("10 张公开资料事实卡（自写摘要）+ 3 份项目文档")
        if st.button("重建索引", help="重新解析 data_sources/ 并重建 Chroma 索引"):
            with st.spinner("正在重建索引…"):
                try:
                    from agent_project.retrieval.ingest import run_ingest

                    counts = run_ingest(reset=True)
                    st.success("完成：" + "，".join(f"{k} {v} 条" for k, v in counts.items()))
                except Exception as exc:  # surfaced to the user, not swallowed
                    st.error(f"重建失败：{exc}")

    # 侧栏收尾：竖排落款 + 朱红小方（印章感）
    st.markdown(
        "<div style='writing-mode:vertical-rl;font-family:KaiTi,STKaiti,serif;font-size:22px;"
        "color:#B5342A;letter-spacing:0.3em;margin:28px auto 6px'>河湟皮影</div>"
        "<div style='width:10px;height:10px;background:#B5342A;margin:0 auto;opacity:0.85'></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main area: conversation first
# ---------------------------------------------------------------------------

render_page_header()

st.caption(
    "对话式非遗顾问：问文化知识（带来源引用）、聊创作想法（多轮补全后调用皮影生成平台）、查生成历史。"
    "生成本体是另一个项目的服务，这里只通过 HTTP 调用。"
)

# 居中收窄单列布局：主流对话产品的标准形态（对话区不铺满宽屏，阅读舒适）
_side_l, center_col, _side_r = st.columns([1, 4, 1], gap="large")

with center_col:
    # Replay history
    if not st.session_state.chat_log:
        # 欢迎页（空状态）：主视觉 + 能力说明 + 可点击示例卡片（2×2），
        # 对齐主流 Agent 产品的 empty-state 范式，对话开始后自动消失
        figure_b64 = _asset_b64("puppet_figure.webp")
        figure_html = (
            f"<img src='data:image/webp;base64,{figure_b64}' style='height:150px'/>"
            if figure_b64 else ""
        )
        st.markdown(
            "<div style='display:flex;flex-direction:column;align-items:center;padding:26px 0 6px'>"
            # 影人立于金线戏台 + 竖排题词
            + "<div style='display:flex;align-items:flex-end;gap:16px'>"
            + figure_html
            + "<div style='writing-mode:vertical-rl;font-family:KaiTi,STKaiti,serif;font-size:18px;"
              "color:#8A2A22;letter-spacing:0.3em;line-height:2'>一盏灯一幕影</div>"
            + "</div>"
            + "<div style='width:230px;border-bottom:3px double #C9A227;margin-top:2px'></div>"
            + "<div style='font-family:KaiTi,STKaiti,serif;font-size:26px;color:#B5342A;"
              "letter-spacing:0.08em;margin-top:14px'>河湟皮影 · 智能创作</div>"
            + "<div style='color:#6b5f4e;font-size:14px;margin-top:4px'>"
              "问文化 · 创皮影 · 查记录 —— 每轮回答皆有出处</div>"
            + "</div>",
            unsafe_allow_html=True,
        )
        cards = [
            ("🏮", "河湟皮影戏是国家级非遗吗？"),
            ("🧩", "皮影影人由哪些部件组成？"),
            ("🎨", "帮我设计一个戴凤冠的女角皮影头像"),
            ("📜", "最近有哪些生成任务？"),
        ]
        for row_i in range(2):
            cols = st.columns(2, gap="medium")
            for col_i, col in enumerate(cols):
                icon, text = cards[row_i * 2 + col_i]
                with col:
                    if st.button(f"{icon}  {text}", key=f"card_{row_i}_{col_i}",
                                 use_container_width=True):
                        st.session_state.pending_input = text
                        st.rerun()
        st.caption("点击卡片直接开始，或在底部输入框自由提问。创作小贴士：描述越具体（角色/冠饰/色彩/纹样），"
                   "补全轮次越少；创作需上传一张侧面头像线稿（全身像/照片不适用）。")
    for entry in st.session_state.chat_log:
        render_entry(entry)

    # Contextual lineart upload: appears only when a creation is underway and the
    # agent is waiting for a lineart — it is a tool input, not the landing feature.
    if st.session_state.slots and not st.session_state.lineart_path:
        with st.expander("📎 创作需要一张头像线稿（侧面头部线稿，全身像/照片不适用）", expanded=True):
            uploaded = st.file_uploader("头像线稿", type=["png", "jpg", "jpeg", "webp", "bmp"], key="inline_upload")
            if uploaded is not None:
                if st.session_state.get("api_mode") and st.session_state.api_session_id:
                    resp = httpx.post(
                        f"{API_BASE}/api/sessions/{st.session_state.api_session_id}/lineart",
                        files={"file": (uploaded.name, uploaded.getbuffer(), uploaded.type or "image/png")},
                        timeout=30.0,
                    )
                    resp.raise_for_status()
                    st.session_state.lineart_path = resp.json()["lineart_path"]
                else:
                    suffix = Path(uploaded.name).suffix.lower()
                    saved = config.UPLOADS_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
                    saved.write_bytes(uploaded.getbuffer())
                    st.session_state.lineart_path = str(saved)
                st.success("线稿已收到。回复“开始生成”即可。")

user_input = st.chat_input("问一个河湟皮影的问题，或描述你想创作的皮影头像…")
if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None

if user_input:
    st.session_state.chat_log.append({"role": "user", "content": user_input})
    with center_col:
        with st.chat_message("user"):
            st.markdown(user_input)

        # Multi-turn context: replay the last few exchanges so follow-ups
        # ("那它的传承人呢？") can be routed and retrieved correctly.
        history = []
        for e in st.session_state.chat_log[-7:-1]:
            history.append(HumanMessage(content=e["content"]) if e["role"] == "user"
                           else AIMessage(content=e["content"]))
        history.append(HumanMessage(content=user_input))

        with st.chat_message("assistant"):
            state_in = {
                "messages": history,
                "slots": st.session_state.slots,
                "lineart_path": st.session_state.lineart_path,
            }
            if st.session_state.get("api_mode") and st.session_state.api_session_id:
                try:
                    result, elapsed = _api_turn(user_input)
                except Exception as exc:
                    st.warning(f"服务接口暂不可用，已切换本地直连模式（{type(exc).__name__}）")
                    st.session_state.api_mode = False
                    result, elapsed = _local_turn(state_in)
            else:
                result, elapsed = _local_turn(state_in)
            entry = {
                "role": "assistant",
                "content": result.get("answer") or "（没有生成回答）",
                "citations": result.get("citations") or [],
                "generation": result.get("generation") or {},
                "trace": result.get("trace") or [],
                "elapsed": elapsed,
            }
            render_generation(entry["generation"])
            render_citations(entry["citations"])
            render_trace(entry["trace"])
            st.caption(f"本轮耗时 {elapsed:.1f}s")

    st.session_state.slots = result.get("slots") or st.session_state.slots
    st.session_state.chat_log.append(entry)
    if entry["generation"].get("final_url") or entry["trace"]:
        st.rerun()  # 侧栏槽位面板即时刷新
