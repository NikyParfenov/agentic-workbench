# Contributing

## Dev setup

```bash
git clone https://github.com/NikyParfenov/agentic-workbench.git
cd agentic-workbench
uv sync --all-extras
```

## Run tests

```bash
uv run pytest
```

## Run type checking

```bash
uv run pyright
```

## Build

```bash
uv build
```

## Adding a new trigger

1. Create `src/agent_runtime_validator/triggers/your_trigger.py`.
2. Subclass `BaseTrigger`, implement `evaluate(trace) -> TriggerResult`.
3. Keep triggers **deterministic** — no LLM calls, no I/O, no side effects.
4. Add the class to `triggers/__init__.py` exports.
5. Add tests in `tests/test_triggers.py`.
6. Update `docs/runtime-validator/triggers.md`.

## Adding a new validator

1. Create `src/agent_runtime_validator/validators/your_validator.py`.
2. Subclass `BaseValidator`, implement `validate(trace, trigger_results)`.
3. Return `ValidatorResult` (sync) or `Awaitable[ValidatorResult]` (async).
4. Add the class to `validators/__init__.py` exports.
5. Add tests in `tests/test_validators.py`.
6. Update `docs/runtime-validator/validators.md`.

## Design principles

- **Triggers are deterministic and cheap.** LLM calls belong in validators.
- **Validators are optional.** The system works with triggers + policy alone.
- **The core is framework-agnostic.** Framework specifics live in `integrations/`.
- **All triggers run on every call.** The pipeline does not short-circuit.
- **Validators only run when a trigger fires.** The healthy path stays fast.
- **Policies are separate from validation.** Validators recommend; policies decide.
