"""Real-service smoke test: QA with evidence + QA at evidence boundary.

Run with env vars set (see start_agent.bat). Not part of the pytest suite —
hits the real LLM and the real vector store.
"""

from langchain_core.messages import HumanMessage

from agent_project.graph.builder import get_graph

QUESTIONS = [
    ("有证据", "河湟皮影戏主要流传在什么地区？它是不是国家级非遗？"),
    ("证据边界", "河湟皮影里的凤冠象征着什么寓意？"),
    ("服务说明", "这个 Agent 如何查看生成任务状态？"),
]

for tag, q in QUESTIONS:
    print(f"\n{'='*60}\n[{tag}] {q}\n{'='*60}")
    out = get_graph().invoke({"messages": [HumanMessage(content=q)]})
    print(f"意图: {out.get('intent')}")
    print(out.get("answer"))
    for c in out.get("citations") or []:
        print(f"  来源【{c['n']}】{c.get('title')}／{c.get('publisher')}")
