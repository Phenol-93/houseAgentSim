"""住宅户型几何判断辅助函数，底层使用 Shapely 实现。"""

from __future__ import annotations

from collections.abc import Sequence

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon


Coordinate = tuple[float, float]
PolygonLike = Polygon | Sequence[Coordinate]


def polygon_from_points(points: Sequence[Coordinate]) -> Polygon:
    """根据户型坐标点创建 Shapely 多边形。"""
    return Polygon(points)


def point_in_polygon(point: Coordinate, polygon: PolygonLike) -> bool:
    """判断点是否位于多边形内部或边界上。"""
    polygon_obj = _ensure_polygon(polygon)
    return polygon_obj.covers(ShapelyPoint(point))


def polygon_area(points: Sequence[Coordinate]) -> float:
    """计算坐标点描述的多边形面积。"""
    return abs(polygon_from_points(points).area)


def is_valid_polygon(points: Sequence[Coordinate]) -> bool:
    """判断坐标点是否构成非空且面积大于 0 的合法多边形。"""
    try:
        polygon = polygon_from_points(points)
    except (TypeError, ValueError):
        return False
    return polygon.is_valid and not polygon.is_empty and polygon.area > 0


def distance_between_points(p1: Coordinate, p2: Coordinate) -> float:
    """计算两个坐标点之间的欧氏距离。"""
    return ShapelyPoint(p1).distance(ShapelyPoint(p2))


def _ensure_polygon(polygon: PolygonLike) -> Polygon:
    if isinstance(polygon, Polygon):
        return polygon
    return polygon_from_points(polygon)


__all__ = [
    "distance_between_points",
    "is_valid_polygon",
    "point_in_polygon",
    "polygon_area",
    "polygon_from_points",
]
