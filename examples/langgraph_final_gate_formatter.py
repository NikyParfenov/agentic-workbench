"""Final-gate validation before a Formatter node — LangGraph example.

Graph shape:
    supervisor / research_agent
           ↓
    ValidationNode(validator_mode="final_gate")
           ├── continue  → formatter → END
           ├── reroute   → supervisor
           └── interrupt / abort → END

No real LLM calls are made. The stub validator always returns valid=True so
the graph walks the happy path on a normal run. Set FORCE_REROUTE=True at the
top to exercise the reroute branch instead.

Run:
    uv run python examples/langgraph_final_gate_formatter.py
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
    formatted_answer: str | None
    cycle_count: int


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def supervisor(state: AgentState) -> AgentState:
    cycle = state.get("cycle_count", 0)
    print(f"[supervisor] cycle={cycle}")
    builder = TraceBuilder.from_trace(state["trace"])
    builder.record_routing("supervisor", "research_agent", reason="dispatch")
    return {
        **state,
        "trace": builder.build(),
        "cycle_count": cycle + 1,
    }


def research_agent(state: AgentState) -> AgentState:
    print("[research_agent] calling lookup_record(demo-record)")
    existing = state["trace"]
    call_id = f"c{len(existing.tool_calls) + 1}"
    builder = (
        TraceBuilder.from_trace(existing)
        .record_tool_call(
            "lookup_record",
            call_id=call_id,
            args={"record_id": "demo-record"},
            agent_name="research_agent",
        )
        .record_tool_result(
            call_id,
            "lookup_record",
            output='{"record_id": "demo-record", "status": "active", "value": 42}',
        )
        .record_artifact(
            artifact_id="result-1",
            artifact_type="json_record",
            content='{"record_id": "demo-record", "status": "active", "value": 42}',
            agent_name="research_agent",
        )
    )
    return {**state, "trace": builder.build()}


def formatter(state: AgentState) -> AgentState:
    print("[formatter] building final answer from trace artifacts")
    artifacts = state["trace"].artifacts
    if artifacts:
        summary = "; ".join(
            f"{a.artifact_id}={a.content[:60]}" for a in artifacts
        )
        answer = f"Formatted answer: {summary}"
    else:
        answer = "Formatted answer: (no artifacts)"
    return {**state, "formatted_answer": answer}


# ---------------------------------------------------------------------------
# Supervisor guard — prevent infinite loops in the demo
# ---------------------------------------------------------------------------

def _supervisor_or_end(state: AgentState) -> str:
    """After supervisor: go to research_agent unless we've hit the cycle cap."""
    if state.get("cycle_count", 0) >= 2:
        print("[supervisor-guard] cycle limit reached → END")
        return "end"
    return "research_agent"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    ts = datetime.now(timezone.utc)

    validation_node = ValidationNode(
        triggers=[SameToolLoopTrigger(max_repeats=5)],
        validator=_StubValidator(),
        validator_mode="final_gate",
        max_validator_calls_per_run=None,
    )

    router = create_validation_router(
        continue_to="formatter",
        reroute_to="supervisor",
        interrupt_to=END,
        abort_to=END,
    )

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor)
    builder.add_node("research_agent", research_agent)
    builder.add_node("validation", validation_node)  # type: ignore[arg-type]
    builder.add_node("formatter", formatter)

    builder.set_entry_point("supervisor")
    builder.add_conditional_edges(
        "supervisor",
        _supervisor_or_end,
        {"research_agent": "research_agent", "end": END},
    )
    builder.add_edge("research_agent", "validation")
    builder.add_conditional_edges("validation", router)
    builder.add_edge("formatter", END)

    return builder.compile(), ts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_demo() -> None:
    graph, ts = build_graph()

    initial: AgentState = {
        "trace": ExecutionTrace(run_id="final-gate-demo", started_at=ts),
        "decision": None,
        "formatted_answer": None,
        "cycle_count": 0,
    }

    print("=== Final-Gate Formatter Demo ===\n")
    final = graph.invoke(initial)

    decision: ValidationDecision | None = final.get("decision")
    print(f"\n[validation] action={decision.action if decision else 'none'}")
    if decision and decision.validator_result:
        print(f"[validation] reason={decision.validator_result.reason}")

    answer = final.get("formatted_answer")
    if answer:
        print(f"\n{answer}")
    else:
        stop_reason = decision.action if decision else "unknown"
        print(f"\nStop reason: {stop_reason}")


if __name__ == "__main__":
    run_demo()
