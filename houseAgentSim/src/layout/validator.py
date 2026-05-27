"""住宅户型几何和引用关系校验工具。"""

from __future__ import annotations

from collections import Counter
from typing import Any

from shapely.geometry import Point as ShapelyPoint

from src.layout.geometry import is_valid_polygon, polygon_from_points
from src.layout.schema import Layout, Wall


LOAD_BEARING_TYPES = {
    "bearing",
    "load bearing",
    "load bearing wall",
    "structural",
    "structural wall",
}

LOAD_BEARING_ID_KEYS = {
    "fixed_walls",
    "load_bearing_wall_ids",
    "load_bearing_walls",
    "non_removable_walls",
    "protected_walls",
    "structural_wall_ids",
    "structural_walls",
}

LOAD_BEARING_GLOBAL_KEYS = {
    "keep_load_bearing_walls",
    "keep_structural_walls",
    "preserve_load_bearing_walls",
    "preserve_structural_walls",
}

EXTERNAL_ROOM_IDS = {"outside", "exterior", "external"}


def validate_layout(layout: Layout) -> list[str]:
    """校验已读取的户型对象，并返回可读的问题说明列表。"""
    issues: list[str] = []

    if layout.grid_size <= 0:
        issues.append("grid_size must be greater than 0.")

    _add_duplicate_id_issues("room", [room.id for room in layout.rooms], issues)
    _add_duplicate_id_issues(
        "furniture", [furniture.id for furniture in layout.furniture], issues
    )
    _add_duplicate_id_issues(
        "activity point",
        [activity_point.id for activity_point in layout.activity_points],
        issues,
    )

    boundary_valid = is_valid_polygon(layout.boundary)
    boundary_polygon = None
    if not boundary_valid:
        issues.append("boundary polygon is invalid.")
    else:
        boundary_polygon = polygon_from_points(layout.boundary)

    rooms_by_id = {room.id: room for room in layout.rooms}
    room_polygons = {}
    for room in layout.rooms:
        if not is_valid_polygon(room.polygon):
            issues.append(f"room '{room.id}' polygon is invalid.")
            continue

        room_polygon = polygon_from_points(room.polygon)
        room_polygons[room.id] = room_polygon
        if boundary_polygon is not None and not boundary_polygon.covers(room_polygon):
            issues.append(f"room '{room.id}' is not fully inside boundary.")

    for furniture in layout.furniture:
        room_polygon = room_polygons.get(furniture.room)
        if furniture.room not in rooms_by_id:
            issues.append(
                f"furniture '{furniture.id}' references unknown room '{furniture.room}'."
            )
            continue
        if room_polygon is None:
            continue
        if not is_valid_polygon(furniture.polygon):
            issues.append(f"furniture '{furniture.id}' polygon is invalid.")
            continue

        furniture_polygon = polygon_from_points(furniture.polygon)
        if not room_polygon.covers(furniture_polygon):
            issues.append(
                f"furniture '{furniture.id}' is not fully inside room '{furniture.room}'."
            )

    for activity_point in layout.activity_points:
        room_polygon = room_polygons.get(activity_point.room)
        if activity_point.room not in rooms_by_id:
            issues.append(
                "activity point "
                f"'{activity_point.id}' references unknown room '{activity_point.room}'."
            )
            continue
        if room_polygon is None:
            continue
        if not room_polygon.covers(ShapelyPoint(activity_point.position)):
            issues.append(
                "activity point "
                f"'{activity_point.id}' is not inside room '{activity_point.room}'."
            )

    room_ids = set(rooms_by_id)
    for door in layout.doors:
        if door.from_room not in room_ids and door.from_room not in EXTERNAL_ROOM_IDS:
            issues.append(
                f"door '{door.id}' from_room references unknown room "
                f"'{door.from_room}'."
            )
        if door.to_room not in room_ids and door.to_room not in EXTERNAL_ROOM_IDS:
            issues.append(
                f"door '{door.id}' to_room references unknown room '{door.to_room}'."
            )

    for wall in layout.walls:
        if _is_load_bearing_wall(wall) and not _wall_is_recognized_by_constraints(
            wall, layout.constraints
        ):
            issues.append(
                f"load-bearing wall '{wall.id}' is not recognizable in constraints."
            )

    return issues


def _add_duplicate_id_issues(label: str, ids: list[str], issues: list[str]) -> None:
    counts = Counter(ids)
    for item_id, count in counts.items():
        if count > 1:
            issues.append(f"duplicate {label} id '{item_id}'.")


def _is_load_bearing_wall(wall: Wall) -> bool:
    normalized = _normalize_wall_type(wall.type)
    return normalized in LOAD_BEARING_TYPES


def _normalize_wall_type(wall_type: str) -> str:
    return wall_type.strip().lower().replace("_", " ").replace("-", " ")


def _wall_is_recognized_by_constraints(wall: Wall, constraints: dict[str, Any]) -> bool:
    for key in LOAD_BEARING_ID_KEYS:
        if key in constraints and _constraint_value_references_wall(
            constraints[key], wall.id
        ):
            return True

    for key in LOAD_BEARING_GLOBAL_KEYS:
        if constraints.get(key) is True:
            return True

    return False


def _constraint_value_references_wall(value: Any, wall_id: str) -> bool:
    if isinstance(value, dict):
        return wall_id in value
    if isinstance(value, (list, tuple, set)):
        return wall_id in value
    return value == wall_id


__all__ = ["validate_layout"]
