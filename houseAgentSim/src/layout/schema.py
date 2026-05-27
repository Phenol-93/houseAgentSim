"""手动标注住宅户型 JSON 的 dataclass 数据结构。

本模块只定义数据结构和字典转换逻辑，不负责文件读取。文件读取请使用
``src.layout.loader``，这样可以保持 schema 层稳定、纯粹、便于复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypeVar


JsonMapping = Mapping[str, Any]
T = TypeVar("T")


@dataclass
class Point:
    x: float
    y: float

    @staticmethod
    def from_dict(data: JsonMapping, path: str = "point") -> "Point":
        _ensure_mapping(data, path)
        return Point(
            x=_as_float(_require(data, "x", path), f"{path}.x"),
            y=_as_float(_require(data, "y", path), f"{path}.y"),
        )


@dataclass
class Room:
    id: str
    name: str
    type: str
    polygon: list[tuple[float, float]]
    capacity: int = 1

    @staticmethod
    def from_dict(data: JsonMapping, path: str = "room") -> "Room":
        _ensure_mapping(data, path)
        return Room(
            id=_as_str(_require(data, "id", path), f"{path}.id"),
            name=_as_str(_require(data, "name", path), f"{path}.name"),
            type=_as_str(_require(data, "type", path), f"{path}.type"),
            polygon=_as_coordinate_list(
                _require(data, "polygon", path), f"{path}.polygon"
            ),
            capacity=_as_int(data.get("capacity", 1), f"{path}.capacity"),
        )


@dataclass
class Wall:
    id: str
    polyline: list[tuple[float, float]]
    type: str

    @staticmethod
    def from_dict(data: JsonMapping, path: str = "wall") -> "Wall":
        _ensure_mapping(data, path)
        return Wall(
            id=_as_str(_require(data, "id", path), f"{path}.id"),
            polyline=_as_coordinate_list(
                _require(data, "polyline", path), f"{path}.polyline"
            ),
            type=_as_str(_require(data, "type", path), f"{path}.type"),
        )


@dataclass
class Door:
    id: str
    from_room: str
    to_room: str
    center: tuple[float, float]
    width: float
    wall_id: str | None = None

    @staticmethod
    def from_dict(data: JsonMapping, path: str = "door") -> "Door":
        _ensure_mapping(data, path)
        return Door(
            id=_as_str(_require(data, "id", path), f"{path}.id"),
            from_room=_as_str(
                _require(data, "from_room", path), f"{path}.from_room"
            ),
            to_room=_as_str(_require(data, "to_room", path), f"{path}.to_room"),
            center=_as_coordinate(_require(data, "center", path), f"{path}.center"),
            width=_as_float(_require(data, "width", path), f"{path}.width"),
            wall_id=_as_optional_str(data.get("wall_id"), f"{path}.wall_id"),
        )


@dataclass
class Furniture:
    id: str
    name: str
    room: str
    polygon: list[tuple[float, float]]
    walkable: bool = False

    @staticmethod
    def from_dict(data: JsonMapping, path: str = "furniture") -> "Furniture":
        _ensure_mapping(data, path)
        return Furniture(
            id=_as_str(_require(data, "id", path), f"{path}.id"),
            name=_as_str(_require(data, "name", path), f"{path}.name"),
            room=_as_str(_require(data, "room", path), f"{path}.room"),
            polygon=_as_coordinate_list(
                _require(data, "polygon", path), f"{path}.polygon"
            ),
            walkable=_as_bool(data.get("walkable", False), f"{path}.walkable"),
        )


@dataclass
class ActivityPoint:
    id: str
    name: str
    room: str
    position: tuple[float, float]
    activity_type: str
    capacity: int = 1

    @staticmethod
    def from_dict(data: JsonMapping, path: str = "activity_point") -> "ActivityPoint":
        _ensure_mapping(data, path)
        return ActivityPoint(
            id=_as_str(_require(data, "id", path), f"{path}.id"),
            name=_as_str(_require(data, "name", path), f"{path}.name"),
            room=_as_str(_require(data, "room", path), f"{path}.room"),
            position=_as_coordinate(
                _require(data, "position", path), f"{path}.position"
            ),
            activity_type=_as_str(
                _require(data, "activity_type", path), f"{path}.activity_type"
            ),
            capacity=_as_int(data.get("capacity", 1), f"{path}.capacity"),
        )


@dataclass
class Layout:
    layout_id: str
    layout_name: str
    unit: str
    grid_size: float
    boundary: list[tuple[float, float]]
    rooms: list[Room]
    walls: list[Wall]
    doors: list[Door]
    furniture: list[Furniture]
    activity_points: list[ActivityPoint]
    constraints: dict[str, Any]

    @staticmethod
    def from_dict(data: JsonMapping, path: str = "layout") -> "Layout":
        _ensure_mapping(data, path)
        return Layout(
            layout_id=_as_str(_require(data, "layout_id", path), f"{path}.layout_id"),
            layout_name=_as_str(
                _require(data, "layout_name", path), f"{path}.layout_name"
            ),
            unit=_as_str(_require(data, "unit", path), f"{path}.unit"),
            grid_size=_as_float(_require(data, "grid_size", path), f"{path}.grid_size"),
            boundary=_as_coordinate_list(
                _require(data, "boundary", path), f"{path}.boundary"
            ),
            rooms=_as_dataclass_list(
                _require(data, "rooms", path), Room.from_dict, f"{path}.rooms"
            ),
            walls=_as_dataclass_list(
                _require(data, "walls", path), Wall.from_dict, f"{path}.walls"
            ),
            doors=_as_dataclass_list(
                _require(data, "doors", path), Door.from_dict, f"{path}.doors"
            ),
            furniture=_as_dataclass_list(
                _require(data, "furniture", path),
                Furniture.from_dict,
                f"{path}.furniture",
            ),
            activity_points=_as_dataclass_list(
                _require(data, "activity_points", path),
                ActivityPoint.from_dict,
                f"{path}.activity_points",
            ),
            constraints=_as_dict(
                _require(data, "constraints", path), f"{path}.constraints"
            ),
        )


def layout_from_dict(data: JsonMapping) -> Layout:
    """把 JSON 字典转换为 ``Layout`` 实例。"""
    return Layout.from_dict(data)


def _require(data: JsonMapping, field: str, path: str) -> Any:
    if field not in data:
        raise ValueError(f"Missing required field: {path}.{field}")
    value = data[field]
    if value is None:
        raise ValueError(f"Required field cannot be null: {path}.{field}")
    return value


def _ensure_mapping(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected object at {path}, got {type(value).__name__}")


def _as_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected string at {path}, got {type(value).__name__}")
    return value


def _as_optional_str(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _as_str(value, path)


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected number at {path}, got {type(value).__name__}")
    return float(value)


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Expected integer at {path}, got {type(value).__name__}")
    return value


def _as_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean at {path}, got {type(value).__name__}")
    return value


def _as_dict(value: Any, path: str) -> dict[str, Any]:
    _ensure_mapping(value, path)
    return dict(value)


def _as_coordinate(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Expected [x, y] coordinate at {path}")
    return (_as_float(value[0], f"{path}[0]"), _as_float(value[1], f"{path}[1]"))


def _as_coordinate_list(value: Any, path: str) -> list[tuple[float, float]]:
    if not isinstance(value, list):
        raise ValueError(f"Expected list of coordinates at {path}")
    return [_as_coordinate(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _as_dataclass_list(
    value: Any,
    factory: Callable[[JsonMapping, str], T],
    path: str,
) -> list[T]:
    if not isinstance(value, list):
        raise ValueError(f"Expected list at {path}, got {type(value).__name__}")
    items: list[T] = []
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        _ensure_mapping(item, item_path)
        items.append(factory(item, item_path))
    return items


__all__ = [
    "ActivityPoint",
    "Door",
    "Furniture",
    "Layout",
    "Point",
    "Room",
    "Wall",
    "layout_from_dict",
]
