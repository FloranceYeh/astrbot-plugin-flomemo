from typing import Any

from astrbot.api import AstrBotConfig

TRUE_VALUES = {"1", "true", "yes", "on"}


class ConfigAccessor:
    def __init__(self, config: AstrBotConfig):
        self._config = config

    def get(self, key: str, default: Any) -> Any:
        return self._config.get(key, default)

    def get_bool(self, key: str, default: bool) -> bool:
        value = self.get(key, default)
        return self._as_bool(value)

    def get_int(
        self,
        key: str,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        value = self.get(key, default)
        return self._as_int(value, default, minimum=minimum, maximum=maximum)

    def get_float(
        self,
        key: str,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        value = self.get(key, default)
        return self._as_float(value, default, minimum=minimum, maximum=maximum)

    def get_group(self, group: str, key: str, default: Any) -> Any:
        container = self.get(group, {})
        if isinstance(container, dict):
            return container.get(key, default)
        return default

    def get_group_bool(self, group: str, key: str, default: bool) -> bool:
        value = self.get_group(group, key, default)
        return self._as_bool(value)

    def get_group_int(
        self,
        group: str,
        key: str,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        value = self.get_group(group, key, default)
        return self._as_int(value, default, minimum=minimum, maximum=maximum)

    def get_group_float(
        self,
        group: str,
        key: str,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        value = self.get_group(group, key, default)
        return self._as_float(value, default, minimum=minimum, maximum=maximum)

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in TRUE_VALUES
        return bool(value)

    def _as_int(
        self,
        value: Any,
        default: int,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            result = default
        if minimum is not None and result < minimum:
            result = minimum
        if maximum is not None and result > maximum:
            result = maximum
        return result

    def _as_float(
        self,
        value: Any,
        default: float,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = default
        if minimum is not None and result < minimum:
            result = minimum
        if maximum is not None and result > maximum:
            result = maximum
        return result
