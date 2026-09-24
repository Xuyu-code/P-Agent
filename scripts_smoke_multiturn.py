"""Real multi-turn smoke: anaphora QA + creation flow + regenerate-with-new-seed.

Simulates exactly what app.py does across turns: carries slots/lineart_path
and message history between graph invocations. Uses the real LLM, real vector
store, and the real generation service. ~2 real generations (~40s GPU total).
"""

import hashlib
import urllib.request

from langchain_core.messages import AIMessage, HumanMessage

from agent_project.graph.builder import get_graph

LINEART = "./tmp_smoke/test_lineart.png"

state = {"slots": None, "lineart_path": None}
messages = []


def turn(text, show_prompt=False):
    messages.append(HumanMessage(content=text))
    out = get_graph().invoke({
        "messages": list(messages),
        "slots": state["slots"],
        "lineart_path": state["lineart_path"],
    })
    state["slots"] = out.get("slots") or state["slots"]
    messages.append(AIMessage(content=out.get("answer", "")))
    print(f"\n{'─'*60}\n👤 {text}\n{'─'*60}")
    print(f"[意图 {out.get('intent')}]")
    for t in out.get("trace", []):
        print(f"  🛠 {t['step']}：{t['detail']}")
    print(out.get("answer"))
    gen = out.get("generation") or {}
    if gen.get("final_url"):
        return gen["final_url"]
    return None


def fetch_md5(url):
    return hashlib.md5(urllib.request.urlopen(url, timeout=15).read()).hexdigest()


# --- 多轮问答（指代追问） ---
turn("河湟皮影戏主要流传在哪里？它是国家级非遗吗？")
turn("那它的代表性传承人呢？")          # 指代：河湟皮影戏
turn("凤冠在河湟皮影里象征什么寓意？")   # 证据边界

# --- 多轮创作 ---
turn("帮我设计一个戴凤冠的女角皮影头像")  # 无线稿 → 应要求上传
state["lineart_path"] = LINEART          # 模拟用户上传
turn("线稿传好了，直接生成")              # 缺色彩，但 skip 短语 → 直接生成
url1 = None
# 上一轮若已生成则拿到 final_url；再要求一版，seed 必须变化
url2 = turn("再生成一版")

print(f"\n{'='*60}\n[seed 校验]")
print("slots:", {k: v for k, v in (state['slots'] or {}).items() if v and k != 'params'})
print("seed:", (state['slots'] or {}).get('params', {}).get('seed'))
if url2:
    print("第二版 final:", url2, "md5:", fetch_md5(url2))
