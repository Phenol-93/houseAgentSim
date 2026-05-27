"""读取家庭成员 JSON，并按需合并手动行为脚本。"""

from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping

from src.agents.behavior import Behavior
from src.agents.resident_agent import ResidentAgent


def load_agents(
    agent_path: str | Path,
    schedule_path: str | Path,
) -> tuple[list[ResidentAgent], list[str]]:
    """读取家庭成员，并为每个成员挂载手动行为日程。"""
    warnings: list[str] = []
    agent_data = _read_json(agent_path, "agent")
    schedule_data = _read_json(schedule_path, "schedule")

    agent_items = _extract_agent_items(agent_data)
    schedules_by_agent = _extract_schedules_by_agent(schedule_data)

    agents: list[ResidentAgent] = []
    for index, item in enumerate(agent_items):
        path = f"agents[{index}]"
        agent_id = _as_str(_require(item, "agent_id", path), f"{path}.agent_id")
        schedule_items = schedules_by_agent.get(agent_id, [])
        if not schedule_items:
            warnings.append(f"agent '{agent_id}' has no schedule.")

        schedule = [
            Behavior.from_dict(behavior, f"schedules.{agent_id}[{behavior_index}]")
            for behavior_index, behavior in enumerate(schedule_items)
        ]

        agents.append(_agent_from_item(item, schedule, path))

    return agents, warnings


def load_agent_profiles(agent_path: str | Path) -> tuple[list[ResidentAgent], list[str]]:
    """只读取家庭成员画像，不挂载手动行为日程。

    该函数用于 AI 行为生成流程：用户仍然负责定义居民画像、性格、习惯和
    初始位置，具体时间表则稍后由模型根据这些画像生成。
    """
    agent_data = _read_json(agent_path, "agent")
    agent_items = _extract_agent_items(agent_data)
    agents = [
        _agent_from_item(item, [], f"agents[{index}]")
        for index, item in enumerate(agent_items)
    ]
    return agents, []


def _read_json(path: str | Path, label: str) -> Any:
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"{label.title()} file does not exist: {json_path}")
    if not json_path.is_file():
        raise ValueError(f"{label.title()} path is not a file: {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} file {json_path}: {exc.msg}") from exc


def _agent_from_item(
    item: Mapping[str, Any],
    schedule: list[Behavior],
    path: str,
) -> ResidentAgent:
    agent_id = _as_str(_require(item, "agent_id", path), f"{path}.agent_id")
    return ResidentAgent(
        agent_id=agent_id,
        name=_as_str(_require(item, "name", path), f"{path}.name"),
        age=_as_int(_require(item, "age", path), f"{path}.age"),
        role=_as_str(_require(item, "role", path), f"{path}.role"),
        mobility=_as_float(_require(item, "mobility", path), f"{path}.mobility"),
        privacy_need=_as_float(
            _require(item, "privacy_need", path), f"{path}.privacy_need"
        ),
        noise_sensitivity=_as_float(
            _require(item, "noise_sensitivity", path), f"{path}.noise_sensitivity"
        ),
        schedule=schedule,
        current_point=_as_str(
            _require(item, "current_point", path), f"{path}.current_point"
        ),
        personality=_as_optional_str(item.get("personality", ""), f"{path}.personality"),
        habits=_as_str_list(item.get("habits", []), f"{path}.habits"),
        needs=_as_str_list(item.get("needs", []), f"{path}.needs"),
        routine_notes=_as_optional_str(
            item.get("routine_notes", ""), f"{path}.routine_notes"
        ),
        profile_extras=_profile_extras(item),
    )


def _extract_agent_items(data: Any) -> list[Mapping[str, Any]]:
    if isinstance(data, list):
        items = data
    elif isinstance(data, Mapping) and isinstance(data.get("agents"), list):
        items = data["agents"]
    else:
        raise ValueError("Agent JSON must be a list or contain an 'agents' list.")

    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"Expected object at agents[{index}]")
        result.append(item)
    return result


def _extract_schedules_by_agent(data: Any) -> dict[str, list[Mapping[str, Any]]]:
    raw_schedules = data.get("schedules") if isinstance(data, Mapping) else data

    if isinstance(raw_schedules, Mapping):
        return _extract_schedule_mapping(raw_schedules)
    if isinstance(raw_schedules, list):
        return _extract_schedule_list(raw_schedules)

    raise ValueError(
        "Schedule JSON must be a list, a mapping, or contain a 'schedules' value."
    )


def _extract_schedule_mapping(
    data: Mapping[str, Any],
) -> dict[str, list[Mapping[str, Any]]]:
    schedules: dict[str, list[Mapping[str, Any]]] = {}
    for agent_id, items in data.items():
        if not isinstance(agent_id, str):
            raise ValueError("Schedule mapping keys must be agent ids as strings.")
        schedules[agent_id] = _as_behavior_item_list(items, f"schedules.{agent_id}")
    return schedules


def _extract_schedule_list(data: list[Any]) -> dict[str, list[Mapping[str, Any]]]:
    schedules: dict[str, list[Mapping[str, Any]]] = {}
    for index, item in enumerate(data):
        path = f"schedules[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"Expected object at {path}")

        agent_id = item.get("agent_id")
        if isinstance(agent_id, str):
            behavior_items = item.get("schedule", item.get("behaviors", []))
            schedules[agent_id] = _as_behavior_item_list(behavior_items, path)
        elif "time" in item:
            raise ValueError(
                "Schedule list entries must include 'agent_id' when using list format."
            )
        else:
            raise ValueError(f"Missing required field: {path}.agent_id")
    return schedules


def _as_behavior_item_list(value: Any, path: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"Expected behavior list at {path}")

    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"Expected object at {path}[{index}]")
        result.append(item)
    return result


def _require(data: Mapping[str, Any], field: str, path: str) -> Any:
    if field not in data:
        raise ValueError(f"Missing required field: {path}.{field}")
    value = data[field]
    if value is None:
        raise ValueError(f"Required field cannot be null: {path}.{field}")
    return value


def _profile_extras(item: Mapping[str, Any]) -> dict[str, Any]:
    known_fields = {
        "agent_id",
        "name",
        "age",
        "role",
        "mobility",
        "privacy_need",
        "noise_sensitivity",
        "schedule",
        "current_point",
        "personality",
        "habits",
        "needs",
        "routine_notes",
    }
    return {str(key): value for key, value in item.items() if key not in known_fields}


def _as_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected string at {path}, got {type(value).__name__}")
    return value


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer at {path}, got {type(value).__name__}")
    return value


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected number at {path}, got {type(value).__name__}")
    return float(value)


def _as_optional_str(value: Any, path: str) -> str:
    if value is None:
        return ""
    return _as_str(value, path)


def _as_str_list(value: Any, path: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"Expected list at {path}, got {type(value).__name__}")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_as_str(item, f"{path}[{index}]"))
    return result


__all__ = ["load_agent_profiles", "load_agents"]
