from typing import Literal

TruncationStrategy = Literal["head", "tail", "middle_ellipsis"]


def truncate(text: str, max_len: int, strategy: TruncationStrategy = "tail") -> str:
    """Truncate text to max_len characters using the given strategy.

    - ``"tail"``            keep beginning, drop end (current default behavior).
    - ``"head"``            drop beginning, keep end (preserves recent content).
    - ``"middle_ellipsis"`` keep both ends, replace middle with an ellipsis marker.
    """
    if len(text) <= max_len:
        return text

    total = len(text)

    if strategy == "tail":
        suffix = f"... ({total} chars total)"
        kept = max_len - len(suffix)
        if kept <= 0:
            return suffix
        return text[:kept] + suffix

    if strategy == "head":
        prefix = f"... ({total} chars total) "
        kept = max_len - len(prefix)
        if kept <= 0:
            return prefix
        return prefix + text[-kept:]

    # middle_ellipsis
    marker = f" ... [{total} chars total] ... "
    half = (max_len - len(marker)) // 2
    if half <= 0:
        return marker
    return text[:half] + marker + text[-half:]
