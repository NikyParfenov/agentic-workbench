from typing import Callable


def apply_redaction(text: str, redact_fn: Callable[[str], str] | None) -> str:
    if redact_fn is None:
        return text
    return redact_fn(text)
