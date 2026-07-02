"""LangGraph example: validator_mode="always" with a ResponseBuilder node.

Graph shape:
    Planner → Analyst → ValidationNode(validator_mode="always")
                            ├── continue  → ResponseBuilder → END
                            ├── reroute   → Planner
                            └── interrupt / abort → Stop

No real LLM calls are made. The stub validator always returns valid=True so
the graph walks the happy path on a normal run. Set FORCE_REROUTE=True at the
top to exercise the reroute branch instead.

Run:
    uv run python examples/langgraph_always_validator.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agent_runtime_validator import (
    ExecutionTrace,
    TraceBuilder,
    ValidatorResult,
    ValidationDecision,
)
from agent_runtime_validator.schema.decisions import TriggerResult
from agent_runtime_validator.validators.base import BaseValidator
from agent_runtime_validator.triggers import SameToolLoopTrigger
from agent_runtime_validator.integrations.langgraph import (
    ValidationNode,
    create_validation_router,
)

# ---------------------------------------------------------------------------
# Toggle to True to exercise the reroute branch (cycle_count cap prevents
# an infinite loop — the second reroute triggers abort via cycle limit).
# ---------------------------------------------------------------------------
FORCE_REROUTE = False


# ---------------------------------------------------------------------------
# Stub validator — deterministic, no LLM
# ---------------------------------------------------------------------------

class _StubValidator(BaseValidator):
    """Always returns valid=True (or reroute when FORCE_REROUTE is set)."""

    def validate(
        self, trace: ExecutionTrace, trigger_results: list[TriggerResult]
    ) -> ValidatorResult:
        if FORCE_REROUTE and trace.metadata.get("_stub_calls", 0) == 0:
            trace.metadata["_stub_calls"] = 1
            return ValidatorResult(
                valid=False,
                confidence=1.0,
                recommendation="reroute",
                reason="stub: forced reroute for demo",
            )
        return ValidatorResult(
            valid=True,
            confidence=1.0,
            recommendation="continue",
            reason="stub: output looks good",
        )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    trace: ExecutionTrace
    decision: ValidationDecision | None
    built_response: str | None
    cycle_count: int


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> AgentState:
    cycle = state.get("cycle_count", 0)
    print(f"[planner] cycle={cycle}")
    builder = TraceBuilder.from_trace(state["trace"])
    builder.record_routing("planner", "analyst", reason="dispatch")
    return {
        **state,
        "trace": builder.build(),
        "cycle_count": cycle + 1,
    }


def analyst_node(state: AgentState) -> AgentState:
    print("[analyst] calling analyze_item(demo-item)")
    existing = state["trace"]
    call_id = f"c{len(existing.tool_calls) + 1}"
    builder = (
        TraceBuilder.from_trace(existing)
        .record_tool_call(
            "analyze_item",
            call_id=call_id,
            args={"item_id": "demo-item"},
            agent_name="analyst",
        )
        .record_tool_result(
            call_id,
            "analyze_item",
            output='{"item_id": "demo-item", "status": "active", "value": 42}',
        )
        .record_artifact(
            artifact_id="result-1",
            artifact_type="json_record",
            content='{"item_id": "demo-item", "status": "active", "value": 42}',
            agent_name="analyst",
        )
    )
    return {**state, "trace": builder.build()}


def response_builder_node(state: AgentState) -> AgentState:
    print("[response_builder] assembling final response from trace artifacts")
    artifacts = state["trace"].artifacts
    if artifacts:
        summary = "; ".join(
            f"{a.artifact_id}={a.content[:60]}" for a in artifacts
        )
        response = f"Built response: {summary}"
    else:
        response = "Built response: (no artifacts)"
    return {**state, "built_response": response}


def stop(state: AgentState) -> AgentState:
    decision: ValidationDecision | None = state.get("decision")
    reason = decision.action if decision else "unknown"
    print(f"[stop] run halted — action={reason}")
    return state


# ---------------------------------------------------------------------------
# Planner guard — prevent infinite loops in the demo
# ---------------------------------------------------------------------------

def _planner_or_end(state: AgentState) -> str:
    """After planner: go to analyst unless we've hit the cycle cap."""
    if state.get("cycle_count", 0) >= 2:
        print("[planner-guard] cycle limit reached → END")
        return "end"
    return "analyst"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    ts = datetime.now(timezone.utc)

    validation_node = ValidationNode(
        triggers=[SameToolLoopTrigger(max_repeats=5)],
        validator=_StubValidator(),
        validator_mode="always",
        max_validator_calls_per_run=None,
    )

    router = create_validation_router(
        continue_to="response_builder",
        reroute_to="planner",
        interrupt_to="stop",
        abort_to="stop",
    )

    builder = StateGraph(AgentState)
    builder.add_node("planner", planner_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("validation", validation_node)  # type: ignore[arg-type]
    builder.add_node("response_builder", response_builder_node)
    builder.add_node("stop", stop)

    builder.set_entry_point("planner")
    builder.add_conditional_edges(
        "planner",
        _planner_or_end,
        {"analyst": "analyst", "end": END},
    )
    builder.add_edge("analyst", "validation")
    builder.add_conditional_edges("validation", router)
    builder.add_edge("response_builder", END)
    builder.add_edge("stop", END)

    return builder.compile(), ts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo() -> None:
    graph, ts = build_graph()

    initial: AgentState = {
        "trace": ExecutionTrace(run_id="always-validator-demo", started_at=ts),
        "decision": None,
        "built_response": None,
        "cycle_count": 0,
    }

    print("=== Always-Validator ResponseBuilder Demo ===\n")
    final = graph.invoke(initial)

    decision: ValidationDecision | None = final.get("decision")
    print(f"\n[validation] action={decision.action if decision else 'none'}")
    if decision and decision.validator_result:
        print(f"[validation] reason={decision.validator_result.reason}")

    response = final.get("built_response")
    if response:
        print(f"\n{response}")
    else:
        stop_reason = decision.action if decision else "unknown"
        print(f"\nStop reason: {stop_reason}")


if __name__ == "__main__":
    run_demo()
