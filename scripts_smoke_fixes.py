"""Focused real re-test of the two fixed defects:
1. anaphora follow-up ("那它的代表性传承人呢？") must surface the inheritor cards;
2. "再生成一版" after "直接生成" must regenerate (seed bumped), not interrogate.
"""

import hashlib
import urllib.request

from langchain_core.messages import AIMessage, HumanMessage

from agent_project.graph.builder import get_graph

LINEART = "./tmp_smoke/test_lineart.png"
state = {"slots": None, "lineart_path": None}
messages = []
final_urls = []


def turn(text):
    messages.append(HumanMessage(content=text))
    out = get_graph().invoke({"messages": list(messages),
                              "slots": state["slots"], "lineart_path": state["lineart_path"]})
    state["slots"] = out.get("slots") or state["slots"]
    messages.append(AIMessage(content=out.get("answer", "")))
    print(f"\n{'─'*60}\n👤 {text}\n{'─'*60}")
    print(out.get("answer"))
    titles = [c.get("title") for c in out.get("citations", [])]
    if titles:
        print("引用:", titles)
    gen = out.get("generation") or {}
    if gen.get("final_url"):
        final_urls.append(gen["final_url"])
    return out


def md5(url):
    return hashlib.md5(urllib.request.urlopen(url, timeout=15).read()).hexdigest()[:10]


turn("河湟皮影戏主要流传在哪里？")
out = turn("那它的代表性传承人呢？")
names = [c.get("title", "") for c in out.get("citations", [])]
hit_inheritor = any("传承人" in t or "周邦辉" in t or "靳生昌" in t for t in names)
print(f"\n[校验1] 传承人卡片被检索引用: {'✅' if hit_inheritor else '❌'} -> {names}")

turn("帮我设计一个戴凤冠的女角皮影头像")
state["lineart_path"] = LINEART
turn("线稿传好了，直接生成")
turn("再生成一版")

print(f"\n[校验2] seed: {(state['slots'] or {}).get('params', {}).get('seed')}（首版 42，再生成应 +1）")
if len(final_urls) == 2:
    m1, m2 = md5(final_urls[0]), md5(final_urls[1])
    print(f"[校验2] 两版图片不同: {'✅' if m1 != m2 else '❌ 相同！'} ({m1} vs {m2})")
    print(final_urls[0])
    print(final_urls[1])
else:
    print(f"[校验2] 只拿到 {len(final_urls)} 张成品图")
