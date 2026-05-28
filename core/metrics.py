import json
from typing import Any

from astrbot.api import logger


def log_metric(name: str, **fields: Any):
    payload = " ".join(
        f"{key}={json.dumps(_normalize_value(value), ensure_ascii=False)}"
        for key, value in fields.items()
        if value is not None
    )
    message = f"[flomemo.metric] {name}"
    if payload:
        message = f"{message} {payload}"
    logger.info(message)


def _normalize_value(value: Any) -> Any:
    """Convert common non-JSON-native values into serializable forms.

    - Keep primitives as-is; convert other objects to str().
    - If value is a set/tuple, convert to list.
    """
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, (set, tuple)):
        return list(value)
    try:
        # Try JSON encoding to see if value is serializable
        json.dumps(value)
        return value
    except Exception:
        try:
            return str(value)
        except Exception:
            return repr(value)
