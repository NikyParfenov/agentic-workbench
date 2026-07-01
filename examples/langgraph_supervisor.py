"""Supervisor-based LangGraph multi-agent demo.

Simulates a Supervisor -> ResearchAgent -> Validation graph. The fake
ResearchAgent repeatedly calls the same tool, which trips:
  - SameToolLoopTrigger
  - NoProgressTrigger

No real LLM calls are made; all agents are fakes. Requires the langgraph extra:
    uv pip install -e ".[langgraph]"

Run:
    uv run python examples/langgraph_supervisor.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import StateGraph, END

from agent_runtime_validator import ValidationDecision
from agent_runtime_validator.schema.trace import ExecutionTrace
from agent_runtime_validator.schema.events import ToolCall, ToolResult, RoutingEvent
from agent_runtime_validator.triggers import (
    MaxRoutesTrigger,
    SameToolLoopTrigger,
    NoProgressTrigger,
)
from agent_runtime_validator.integrations.langgraph.nodes import ValidationNode


class AgentState(TypedDict):
    messages: list[str]
    trace: ExecutionTrace
    decision: ValidationDecision | None
    step: int


def supervisor_node(state: AgentState) -> AgentState:
    decision = state.get("decision")
    if decision and not decision.should_continue:
        return {**state, "messages": state["messages"] + [f"[Supervisor] Stopping: {decision.action}"]}

    step = state.get("step", 0)
    ts = datetime.now(timezone.utc)
    routing = RoutingEvent(from_agent="Supervisor", to_agent="ResearchAgent", timestamp=ts)
    trace = state["trace"]
    updated_trace = trace.model_copy(
        update={"routing_events": trace.routing_events + [routing]}
    )
    return {
        **state,
        "trace": updated_trace,
        "messages": state["messages"] + ["[Supervisor] Routing to ResearchAgent"],
        "step": step + 1,
    }


def research_agent_node(state: AgentState) -> AgentState:
    ts = datetime.now(timezone.utc)
    call_id = f"c{len(state['trace'].tool_calls) + 1}"
    call = ToolCall(tool_name="lookup_record", call_id=call_id, args={"record_id": "demo-record"}, timestamp=ts)
    result = ToolResult(call_id=call_id, tool_name="lookup_record", output="not found", timestamp=ts)
    trace = state["trace"]
    updated_trace = trace.model_copy(update={
        "tool_calls": trace.tool_calls + [call],
        "tool_results": trace.tool_results + [result],
    })
    return {
        **state,
        "trace": updated_trace,
        "messages": state["messages"] + ["[ResearchAgent] Called lookup_record(demo-record) -> not found"],
    }


def should_continue(state: AgentState) -> str:
    decision = state.get("decision")
    if decision and not decision.should_continue:
        return "end"
    step = state.get("step", 0)
    if step >= 6:
        return "end"
    return "research_agent"


def build_graph():
    validation_node = ValidationNode(
        triggers=[
            MaxRoutesTrigger(max_routes=10),
            SameToolLoopTrigger(max_repeats=3),
            NoProgressTrigger(min_tool_calls=3),
        ],
    )

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research_agent", research_agent_node)
    # ValidationNode is a plain callable; langgraph's Node protocol is stricter.
    builder.add_node("validation", validation_node)  # type: ignore[arg-type]

    builder.set_entry_point("supervisor")
    builder.add_conditional_edges("supervisor", should_continue, {
        "research_agent": "research_agent",
        "end": END,
    })
    builder.add_edge("research_agent", "validation")
    builder.add_edge("validation", "supervisor")

    return builder.compile()


def run_demo() -> None:
    graph = build_graph()

    ts = datetime.now(timezone.utc)
    initial_state: AgentState = {
        "messages": ["[User] Look up the demo-record entry"],
        "trace": ExecutionTrace(run_id="demo-supervisor", started_at=ts),
        "decision": None,
        "step": 0,
    }

    print("=== Supervisor Demo ===\n")
    final = graph.invoke(initial_state)

    print("--- Messages ---")
    for msg in final["messages"]:
        print(msg)

    print("\n--- Final Validation Decision ---")
    decision = final.get("decision")
    if decision:
        print(f"Action:    {decision.action}")
        print(f"Severity:  {decision.severity}")
        print(f"Triggered: {decision.triggered_by}")
        print(f"Reason:    {decision.reason}")
    else:
        print("No validation triggered")


if __name__ == "__main__":
    run_demo()
