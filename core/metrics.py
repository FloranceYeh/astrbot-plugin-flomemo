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
