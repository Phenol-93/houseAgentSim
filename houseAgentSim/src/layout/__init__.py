"""户型读取、数据结构、几何工具和校验接口。"""

from src.layout.loader import load_layout, save_layout
from src.layout.geometry import (
    distance_between_points,
    is_valid_polygon,
    point_in_polygon,
    polygon_area,
    polygon_from_points,
)
from src.layout.schema import (
    ActivityPoint,
    Door,
    Furniture,
    Layout,
    Point,
    Room,
    Wall,
    layout_from_dict,
)
from src.layout.validator import validate_layout

__all__ = [
    "ActivityPoint",
    "Door",
    "Furniture",
    "Layout",
    "Point",
    "Room",
    "Wall",
    "distance_between_points",
    "is_valid_polygon",
    "layout_from_dict",
    "load_layout",
    "point_in_polygon",
    "polygon_area",
    "polygon_from_points",
    "save_layout",
    "validate_layout",
]
