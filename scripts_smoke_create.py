"""Real-service smoke: one real creation through the external service.

Requires: env vars (start_agent.bat), index built, and the generation
service running at PUPPET_API_BASE.
"""

from langchain_core.messages import HumanMessage

from agent_project.graph.builder import get_graph

LINEART = "./tmp_smoke/test_lineart.png"

# Real creation through the generation service
q = "帮我生成一个戴凤冠、右侧面、暖黄色皮影风格的女性头像"
print(f"\n{'='*60}\n[真实创作] {q}\n{'='*60}")
out = get_graph().invoke({"messages": [HumanMessage(content=q)], "lineart_path": LINEART})
print(f"意图: {out.get('intent')}")
print(out.get("answer"))
gen = out.get("generation") or {}
print("prompt:", gen.get("prompt"))
print("final_url:", gen.get("final_url"))
