"""把手动标注的户型 JSON 转换为可寻路的网格模型。

网格生成是整个模拟的空间基础：房间决定网格所属区域，家具和墙体决定
是否可通行，门洞负责重新打开墙体上的通行位置，活动点则把居民行为目标
映射到具体网格坐标。
"""

from __future__ import annotations

import math

from shapely.geometry import LineString
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import box

from src.grid.grid_cell import GridCell
from src.grid.grid_model import Grid
from src.layout.geometry import polygon_from_points
from src.layout.schema import Door, Layout, Wall


DEFAULT_CELL_COST = 1.0
WALKABLE_FURNITURE_COST = 2.0
DEFAULT_WALL_THICKNESS = 0.12


def build_grid(layout: Layout) -> tuple[Grid, list[str]]:
    """根据户型边界、家具、墙体、门洞和活动点生成 ``Grid``。"""
    warnings: list[str] = []

    if layout.grid_size <= 0:
        raise ValueError("layout.grid_size must be greater than 0 to build a grid.")

    boundary_polygon = polygon_from_points(layout.boundary)
    min_x, min_y, max_x, max_y = boundary_polygon.bounds
    width = max(1, math.ceil((max_x - min_x) / layout.grid_size))
    height = max(1, math.ceil((max_y - min_y) / layout.grid_size))
    origin = (float(min_x), float(min_y))

    # 预先转换 Shapely 几何，后续逐格判断时可以复用，减少重复构造对象。
    room_polygons = [
        (room.id, polygon_from_points(room.polygon)) for room in layout.rooms
    ]
    furniture_polygons = [
        (furniture, polygon_from_points(furniture.polygon))
        for furniture in layout.furniture
    ]
    wall_lines = _wall_lines(layout)

    # 采用“网格中心点采样”：中心点在户型边界内，则认为该格属于室内。
    cells: list[list[GridCell]] = []
    for row in range(height):
        row_cells: list[GridCell] = []
        for col in range(width):
            center = _grid_to_world(row, col, origin, layout.grid_size)
            center_point = ShapelyPoint(center)
            inside_boundary = boundary_polygon.covers(center_point)
            room_id = _find_room_id(center_point, room_polygons)
            cell = GridCell(
                row=row,
                col=col,
                center=center,
                walkable=inside_boundary,
                room_id=room_id,
                blocked=not inside_boundary,
                block_type="outside_boundary" if not inside_boundary else None,
                cost=DEFAULT_CELL_COST,
                tags=set(),
            )

            if inside_boundary:
                _apply_furniture_to_cell(cell, center_point, furniture_polygons)

            row_cells.append(cell)
        cells.append(row_cells)

    grid = Grid(
        cells=cells,
        grid_size=layout.grid_size,
        width=width,
        height=height,
        origin=origin,
    )

    _apply_wall_and_door_handling(
        layout,
        grid,
        wall_lines,
        boundary_polygon.boundary,
        warnings,
    )
    _map_activity_points(layout, grid, warnings)

    return grid, warnings


def _grid_to_world(
    row: int,
    col: int,
    origin: tuple[float, float],
    grid_size: float,
) -> tuple[float, float]:
    origin_x, origin_y = origin
    return (
        origin_x + (col + 0.5) * grid_size,
        origin_y + (row + 0.5) * grid_size,
    )


def _find_room_id(center_point: ShapelyPoint, room_polygons) -> str | None:
    for room_id, room_polygon in room_polygons:
        if room_polygon.covers(center_point):
            return room_id
    return None


def _apply_furniture_to_cell(
    cell: GridCell,
    center_point: ShapelyPoint,
    furniture_polygons,
) -> None:
    """根据家具多边形更新网格通行状态和通行成本。"""
    for furniture, furniture_polygon in furniture_polygons:
        if not furniture_polygon.covers(center_point):
            continue

        cell.tags.add("furniture")
        cell.tags.add(f"furniture:{furniture.id}")
        if furniture.walkable:
            cell.cost = max(cell.cost, WALKABLE_FURNITURE_COST)
            cell.tags.add("walkable_furniture")
        else:
            cell.blocked = True
            cell.walkable = False
            cell.block_type = "furniture"
            cell.tags.add("blocked_by_furniture")


def _map_activity_points(layout: Layout, grid: Grid, warnings: list[str]) -> None:
    """把行为目标点映射到网格坐标，供模拟和寻路直接调用。"""
    for activity_point in layout.activity_points:
        row, col = grid.world_to_grid(*activity_point.position)
        if not grid.in_bounds(row, col):
            warnings.append(
                f"activity point '{activity_point.id}' maps outside grid bounds."
            )
            continue

        grid.set_activity_point(activity_point.id, row, col)
        cell = grid.get_cell(row, col)
        if cell is not None and cell.blocked:
            warnings.append(
                f"activity point '{activity_point.id}' falls on blocked cell "
                f"({row}, {col})."
            )


def _apply_wall_and_door_handling(
    layout: Layout,
    grid: Grid,
    wall_lines: dict[str, LineString],
    boundary_line,
    warnings: list[str],
) -> None:
    """先把室内墙体栅格化为障碍，再按门洞位置恢复通行。"""
    wall_geometries = _wall_block_geometries(layout, wall_lines, boundary_line)
    if not wall_geometries:
        return

    for row in range(grid.height):
        for col in range(grid.width):
            cell = grid.get_cell(row, col)
            if cell is None or not cell.walkable or cell.block_type == "outside_boundary":
                continue

            cell_polygon = _cell_polygon(cell.center, grid.grid_size)
            for wall, wall_geometry in wall_geometries:
                if not wall_geometry.intersects(cell_polygon):
                    continue
                cell.tags.add("wall")
                cell.tags.add(f"wall:{wall.id}")
                if cell.block_type != "furniture":
                    cell.blocked = True
                    cell.walkable = False
                    cell.block_type = "wall"
                    cell.tags.add("blocked_by_wall")
                break

    door_geometries = _door_opening_geometries(layout, wall_lines, grid, warnings)
    for door, door_geometry in door_geometries:
        for row in range(grid.height):
            for col in range(grid.width):
                cell = grid.get_cell(row, col)
                if cell is None or cell.block_type != "wall":
                    continue
                if not door_geometry.intersects(_cell_polygon(cell.center, grid.grid_size)):
                    continue
                cell.blocked = False
                cell.walkable = True
                cell.block_type = None
                cell.tags.discard("blocked_by_wall")
                cell.tags.add("door_opening")
                cell.tags.add(f"door:{door.id}")


def _wall_lines(layout: Layout) -> dict[str, LineString]:
    lines: dict[str, LineString] = {}
    for wall in layout.walls:
        if len(wall.polyline) < 2:
            continue
        line = LineString(wall.polyline)
        if not line.is_empty:
            lines[wall.id] = line
    return lines


def _wall_block_geometries(
    layout: Layout,
    wall_lines: dict[str, LineString],
    boundary_line,
) -> list[tuple[Wall, object]]:
    """生成墙体阻挡带；外墙由户型边界处理，这里只处理室内墙。"""
    wall_half_thickness = _wall_half_thickness(layout)
    geometries = []
    for wall in layout.walls:
        line = wall_lines.get(wall.id)
        if line is None or _is_exterior_wall(line, boundary_line, layout.grid_size):
            continue
        geometries.append(
            (
                wall,
                line.buffer(
                    wall_half_thickness,
                    cap_style="square",
                    join_style="mitre",
                ),
            )
        )
    return geometries


def _door_opening_geometries(
    layout: Layout,
    wall_lines: dict[str, LineString],
    grid: Grid,
    warnings: list[str],
) -> list[tuple[Door, object]]:
    """根据门洞中心、宽度和所在墙线生成可通行开口区域。"""
    geometries = []
    door_clearance = _door_clearance(layout, grid)
    for door in layout.doors:
        wall_line = _door_wall_line(door, wall_lines)
        if wall_line is None:
            warnings.append(
                f"door '{door.id}' is not close to any wall; no wall opening was applied."
            )
            continue

        opening_line = _door_opening_line(door, wall_line)
        if opening_line is None:
            warnings.append(
                f"door '{door.id}' could not infer an opening direction from its wall."
            )
            continue

        geometries.append(
            (
                door,
                opening_line.buffer(
                    door_clearance,
                    cap_style="square",
                    join_style="mitre",
                ),
            )
        )
    return geometries


def _door_wall_line(door: Door, wall_lines: dict[str, LineString]) -> LineString | None:
    if door.wall_id:
        return wall_lines.get(door.wall_id)

    center_point = ShapelyPoint(door.center)
    nearest_line = None
    nearest_distance = math.inf
    for line in wall_lines.values():
        distance = line.distance(center_point)
        if distance < nearest_distance:
            nearest_line = line
            nearest_distance = distance
    return nearest_line


def _door_opening_line(door: Door, wall_line: LineString) -> LineString | None:
    center_point = ShapelyPoint(door.center)
    projected_distance = wall_line.project(center_point)
    epsilon = max(door.width / 2, 0.01)
    start_distance = max(0.0, projected_distance - epsilon)
    end_distance = min(wall_line.length, projected_distance + epsilon)
    if end_distance <= start_distance:
        return None

    start = wall_line.interpolate(start_distance)
    end = wall_line.interpolate(end_distance)
    return LineString([(start.x, start.y), (end.x, end.y)])


def _cell_polygon(center: tuple[float, float], grid_size: float):
    half_size = grid_size / 2
    return box(
        center[0] - half_size,
        center[1] - half_size,
        center[0] + half_size,
        center[1] + half_size,
    )


def _is_exterior_wall(
    line: LineString,
    boundary_line,
    grid_size: float,
) -> bool:
    tolerance = max(grid_size * 0.05, 1e-6)
    return line.difference(boundary_line.buffer(tolerance)).is_empty


def _wall_half_thickness(layout: Layout) -> float:
    wall_thickness = layout.constraints.get("wall_thickness")
    if isinstance(wall_thickness, (int, float)) and wall_thickness > 0:
        return float(wall_thickness) / 2
    return min(DEFAULT_WALL_THICKNESS / 2, layout.grid_size * 0.45)


def _door_clearance(layout: Layout, grid: Grid) -> float:
    wall_half_thickness = _wall_half_thickness(layout)
    return max(wall_half_thickness + grid.grid_size * 0.51, grid.grid_size * 0.55)


__all__ = ["build_grid"]
