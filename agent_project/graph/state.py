"""LangGraph conversation state.

`slots` / `lineart_path` persist across turns (the Streamlit app keeps them in
session state and feeds them back in), which is how multi-turn slot filling
works without LangGraph interrupts in the MVP.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    intent: str                      # qa | create | history | chat
    lineart_path: str | None         # uploaded lineart available to this session
    slots: dict[str, Any]            # serialized CreationSlots, carried across turns
    evidence: list[dict[str, Any]]   # retrieval hits used for the QA answer
    answer: str                      # final assistant text for this turn
    citations: list[dict[str, Any]]  # sources backing the answer
    generation: dict[str, Any]       # task_id / status / result of a creation run
    error: str | None
    trace: list[dict[str, str]]      # 本轮处理过程（意图/检索/工具调用/状态），供界面展示
