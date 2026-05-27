"""住宅行为模拟日志的空间冲突检测工具。"""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from typing import Any, Iterable

from src.layout.schema import ActivityPoint, Layout


BATHROOM_KEYWORDS = {
    "bath",
    "bathroom",
    "toilet",
    "wc",
    "washroom",
    "restroom",
    "卫生",
    "厕所",
    "洗手",
    "浴",
}
WORK_KEYWORDS = {"work", "desk", "meeting", "focus", "study", "办公", "工作", "书桌"}
VISITOR_KEYWORDS = {"visitor", "guest", "访客", "客人"}
NIGHT_START = 22 * 60
NIGHT_END = 6 * 60


def detect_bathroom_queue(wait_log, activity_points) -> list[dict]:
    """检测指向卫生间相关活动点的等待事件。"""
    bathroom_point_ids = _bathroom_activity_point_ids(activity_points)
    events: list[dict] = []

    for record in wait_log:
        target = str(record.get("target", ""))
        reason = str(record.get("reason", ""))
        if target not in bathroom_point_ids and not _contains_keyword(
            f"{target} {reason}", BATHROOM_KEYWORDS
        ):
            continue

        events.append(
            {
                "conflict_type": "bathroom_queue",
                "time": record.get("time"),
                "agent_id": record.get("agent_id"),
                "target": target,
                "reason": reason,
                "involved_agents": _compact_list([record.get("agent_id")]),
                "description": f"Agent waited for bathroom target '{target}'.",
            }
        )

    return events


def detect_work_interference(
    path_log,
    occupancy_log,
    work_point_ids,
    buffer_distance,
) -> list[dict]:
    """检测同一时间步内路径是否靠近已被占用的工作点。"""
    work_point_id_set = set(work_point_ids)
    work_positions = _infer_activity_point_grids(path_log, occupancy_log, work_point_id_set)
    work_occupancy_by_time = _work_occupancy_by_time(occupancy_log, work_point_id_set)
    events: list[dict] = []

    for path_record in path_log:
        time = path_record.get("time")
        path = _as_grid_path(path_record.get("path", []))
        if not time or not path:
            continue

        moving_agent_id = path_record.get("agent_id")
        for occupancy in work_occupancy_by_time.get(time, []):
            work_point = occupancy.get("point")
            occupant_id = occupancy.get("agent_id")
            if occupant_id == moving_agent_id:
                continue

            work_grid = _grid_coord_from_record(occupancy) or work_positions.get(work_point)
            if work_grid is None:
                continue

            nearest_distance = _min_grid_distance(path, work_grid)
            if nearest_distance <= buffer_distance:
                events.append(
                    {
                        "conflict_type": "work_interference",
                        "time": time,
                        "agent_id": moving_agent_id,
                        "target": path_record.get("target_point"),
                        "work_point": work_point,
                        "work_agent_id": occupant_id,
                        "distance": nearest_distance,
                        "involved_agents": _compact_list([moving_agent_id, occupant_id]),
                        "description": (
                            f"Path passed within {nearest_distance:.2f} grid cells "
                            f"of occupied work point '{work_point}'."
                        ),
                    }
                )

    return events


def detect_privacy_exposure(path_log, layout: Layout, private_room_ids) -> list[dict]:
    """检测访客路径是否经过私密房间。"""
    private_room_id_set = set(private_room_ids)
    activity_point_rooms = {
        activity_point.id: activity_point.room for activity_point in layout.activity_points
    }
    events: list[dict] = []

    for record in path_log:
        if not _is_visitor_path(record):
            continue

        crossed_rooms = _rooms_from_path_record(record)
        if not crossed_rooms:
            crossed_rooms = _rooms_from_activity_points(record, activity_point_rooms)

        exposed_rooms = [
            room_id for room_id in crossed_rooms if room_id in private_room_id_set
        ]
        if not exposed_rooms:
            continue

        events.append(
            {
                "conflict_type": "privacy_exposure",
                "time": record.get("time"),
                "agent_id": record.get("agent_id"),
                "target": record.get("target_point"),
                "private_rooms": exposed_rooms,
                "involved_agents": _compact_list([record.get("agent_id")]),
                "description": (
                    "Visitor path exposed private room(s): "
                    + ", ".join(exposed_rooms)
                    + "."
                ),
            }
        )

    return events


def detect_elderly_night_risk(path_log, elderly_agent_id) -> list[dict]:
    """检测老人夜间前往卫生间的风险事件。"""
    events: list[dict] = []

    for record in path_log:
        if record.get("agent_id") != elderly_agent_id:
            continue
        if not _is_night_time(str(record.get("time", ""))):
            continue

        intent_text = " ".join(
            str(record.get(field, ""))
            for field in ("activity", "from_point", "target_point")
        )
        if not _contains_keyword(intent_text, BATHROOM_KEYWORDS):
            continue

        events.append(
            {
                "conflict_type": "elderly_night_risk",
                "time": record.get("time"),
                "agent_id": elderly_agent_id,
                "target": record.get("target_point"),
                "path_length": record.get("path_length", 0),
                "turn_count": record.get("turn_count", 0),
                "involved_agents": [elderly_agent_id],
                "description": (
                    f"Elderly agent '{elderly_agent_id}' moved to bathroom at night."
                ),
            }
        )

    return events


def summarize_conflicts(conflict_events) -> list[dict]:
    """按冲突类型汇总事件数量、时间和涉及成员。"""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for event in conflict_events:
        conflict_type = event.get("conflict_type", event.get("type", "unknown"))
        grouped[str(conflict_type)].append(event)

    summaries: list[dict] = []
    for conflict_type, events in grouped.items():
        times = sorted(
            time for time in (event.get("time") for event in events) if time is not None
        )
        involved_agents: set[str] = set()
        for event in events:
            for agent_id in _event_agents(event):
                involved_agents.add(agent_id)

        summaries.append(
            {
                "conflict_type": conflict_type,
                "count": len(events),
                "time_start": times[0] if times else None,
                "time_end": times[-1] if times else None,
                "times": times,
                "involved_agents": sorted(involved_agents),
            }
        )

    return sorted(summaries, key=lambda item: item["conflict_type"])


def _bathroom_activity_point_ids(activity_points) -> set[str]:
    point_ids: set[str] = set()
    for activity_point in activity_points:
        point_id = _get_attr_or_key(activity_point, "id")
        text = " ".join(
            str(_get_attr_or_key(activity_point, field, ""))
            for field in ("id", "name", "room", "activity_type")
        )
        if point_id and _contains_keyword(text, BATHROOM_KEYWORDS):
            point_ids.add(str(point_id))
    return point_ids


def _infer_activity_point_grids(
    path_log,
    occupancy_log,
    point_ids: set[str],
) -> dict[str, tuple[int, int]]:
    positions: dict[str, tuple[int, int]] = {}

    for record in path_log:
        from_point = record.get("from_point")
        target_point = record.get("target_point")
        if from_point in point_ids:
            coord = _as_grid_coord(record.get("start_grid"))
            if coord is not None:
                positions[str(from_point)] = coord
        if target_point in point_ids:
            coord = _as_grid_coord(record.get("goal_grid"))
            if coord is not None:
                positions[str(target_point)] = coord

    for record in occupancy_log:
        point = record.get("point")
        coord = _grid_coord_from_record(record)
        if point in point_ids and coord is not None:
            positions[str(point)] = coord

    return positions


def _work_occupancy_by_time(occupancy_log, work_point_ids: set[str]) -> dict[str, list[dict]]:
    by_time: dict[str, list[dict]] = defaultdict(list)
    for record in occupancy_log:
        point = record.get("point")
        activity = str(record.get("activity", ""))
        if point in work_point_ids or _contains_keyword(activity, WORK_KEYWORDS):
            by_time[str(record.get("time"))].append(record)
    return by_time


def _grid_coord_from_record(record: dict) -> tuple[int, int] | None:
    for key in ("grid", "point_grid", "current_grid", "position_grid"):
        coord = _as_grid_coord(record.get(key))
        if coord is not None:
            return coord
    return None


def _as_grid_path(value: Any) -> list[tuple[int, int]]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(value, list):
        return []

    path: list[tuple[int, int]] = []
    for item in value:
        coord = _as_grid_coord(item)
        if coord is not None:
            path.append(coord)
    return path


def _as_grid_coord(value: Any) -> tuple[int, int] | None:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(isinstance(item, (int, float)) for item in value):
        return None
    return int(value[0]), int(value[1])


def _min_grid_distance(
    path: list[tuple[int, int]],
    target: tuple[int, int],
) -> float:
    return min(math.dist(coord, target) for coord in path)


def _is_visitor_path(record: dict) -> bool:
    text = " ".join(
        str(record.get(field, ""))
        for field in ("agent_id", "agent_name", "agent_role", "role", "activity")
    )
    return _contains_keyword(text, VISITOR_KEYWORDS)


def _rooms_from_path_record(record: dict) -> list[str]:
    for key in ("rooms", "room_ids", "path_rooms", "crossed_rooms"):
        rooms = record.get(key)
        if isinstance(rooms, str):
            try:
                rooms = ast.literal_eval(rooms)
            except (SyntaxError, ValueError):
                rooms = [rooms]
        if isinstance(rooms, list):
            return [str(room_id) for room_id in rooms]
    return []


def _rooms_from_activity_points(
    record: dict,
    activity_point_rooms: dict[str, str],
) -> list[str]:
    rooms: list[str] = []
    for key in ("from_point", "target_point"):
        point = record.get(key)
        room = activity_point_rooms.get(str(point))
        if room is not None and room not in rooms:
            rooms.append(room)
    return rooms


def _is_night_time(time: str) -> bool:
    minutes = _time_to_minutes_or_none(time)
    if minutes is None:
        return False
    return minutes >= NIGHT_START or minutes < NIGHT_END


def _time_to_minutes_or_none(time: str) -> int | None:
    parts = time.split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    if hour == 24 and minute == 0:
        return 24 * 60
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    return hour * 60 + minute


def _event_agents(event: dict) -> list[str]:
    agents = event.get("involved_agents")
    if isinstance(agents, list):
        return [str(agent_id) for agent_id in agents if agent_id is not None]
    return _compact_list([event.get("agent_id"), event.get("work_agent_id")])


def _compact_list(values: Iterable[Any]) -> list[str]:
    return [str(value) for value in values if value not in (None, "")]


def _contains_keyword(text: str, keywords: set[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _get_attr_or_key(value: ActivityPoint | dict, key: str, default=None):
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


__all__ = [
    "detect_bathroom_queue",
    "detect_elderly_night_risk",
    "detect_privacy_exposure",
    "detect_work_interference",
    "summarize_conflicts",
]
