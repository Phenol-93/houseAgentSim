"""住户智能体的一条行为日程。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class Behavior:
    time: str
    activity: str
    target: str
    duration: int
    tags: list[str] = field(default_factory=list)
    priority: int = 1

    @staticmethod
    def from_dict(data: Mapping[str, Any], path: str = "behavior") -> "Behavior":
        """从 JSON 字典创建一条行为日程。"""
        if not isinstance(data, Mapping):
            raise ValueError(f"Expected object at {path}, got {type(data).__name__}")

        return Behavior(
            time=_as_str(_require(data, "time", path), f"{path}.time"),
            activity=_as_str(_require(data, "activity", path), f"{path}.activity"),
            target=_as_str(_require(data, "target", path), f"{path}.target"),
            duration=_as_int(_require(data, "duration", path), f"{path}.duration"),
            tags=_as_str_list(data.get("tags", []), f"{path}.tags"),
            priority=_as_int(data.get("priority", 1), f"{path}.priority"),
        )


def _require(data: Mapping[str, Any], field: str, path: str) -> Any:
    if field not in data:
        raise ValueError(f"Missing required field: {path}.{field}")
    value = data[field]
    if value is None:
        raise ValueError(f"Required field cannot be null: {path}.{field}")
    return value


def _as_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected string at {path}, got {type(value).__name__}")
    return value


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer at {path}, got {type(value).__name__}")
    return value


def _as_str_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Expected list at {path}, got {type(value).__name__}")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_as_str(item, f"{path}[{index}]"))
    return result


__all__ = ["Behavior"]
