"""网格路径分析辅助函数。"""

from __future__ import annotations

import math
from collections.abc import Iterable

from src.grid.grid_model import Grid


GridCoord = tuple[int, int]


def compute_path_length(path: list[GridCoord], grid_size: float) -> float:
    """根据网格坐标计算路径几何长度。"""
    if len(path) < 2:
        return 0.0

    total = 0.0
    for current, following in zip(path, path[1:]):
        row_delta = abs(following[0] - current[0])
        col_delta = abs(following[1] - current[1])
        total += math.hypot(row_delta, col_delta) * grid_size
    return total


def compute_turn_count(path: list[GridCoord]) -> int:
    """统计路径中的方向变化次数。"""
    if len(path) < 3:
        return 0

    turns = 0
    previous_direction = _step_direction(path[0], path[1])
    for current, following in zip(path[1:], path[2:]):
        direction = _step_direction(current, following)
        if direction != previous_direction:
            turns += 1
        previous_direction = direction
    return turns


def expand_path_cells(path: list[GridCoord]) -> list[GridCoord]:
    """把稀疏路径节点展开为每段线段经过的网格。"""
    if len(path) < 2:
        return list(path)

    expanded: list[GridCoord] = []
    for start, end in zip(path, path[1:]):
        segment = grid_line_cells(start, end)
        if expanded and segment and segment[0] == expanded[-1]:
            expanded.extend(segment[1:])
        else:
            expanded.extend(segment)
    return expanded


def grid_line_cells(start: GridCoord, end: GridCoord) -> list[GridCoord]:
    """返回中心点连线经过的 Bresenham 网格序列。"""
    row0, col0 = start
    row1, col1 = end
    delta_col = abs(col1 - col0)
    delta_row = abs(row1 - row0)
    step_col = 1 if col0 < col1 else -1
    step_row = 1 if row0 < row1 else -1

    cells: list[GridCoord] = []
    error = delta_col - delta_row
    row = row0
    col = col0

    while True:
        cells.append((row, col))
        if row == row1 and col == col1:
            break
        doubled_error = 2 * error
        if doubled_error > -delta_row:
            error -= delta_row
            col += step_col
        if doubled_error < delta_col:
            error += delta_col
            row += step_row
    return cells


def path_crosses_room(path: list[GridCoord], grid: Grid) -> list[str]:
    """返回路径经过的房间 id，并保留首次出现顺序。"""
    room_ids: list[str] = []
    seen: set[str] = set()
    for row, col in expand_path_cells(path):
        cell = grid.get_cell(row, col)
        if cell is None or cell.room_id is None or cell.room_id in seen:
            continue
        seen.add(cell.room_id)
        room_ids.append(cell.room_id)
    return room_ids


def path_near_furniture(
    path: list[GridCoord],
    grid: Grid,
    furniture_ids: Iterable[str],
) -> list[str]:
    """返回路径及其 8 邻域附近出现的家具 id。"""
    furniture_id_set = set(furniture_ids)
    nearby: list[str] = []
    seen: set[str] = set()

    for row, col in expand_path_cells(path):
        cells = []
        cell = grid.get_cell(row, col)
        if cell is not None:
            cells.append(cell)
        cells.extend(grid.neighbors_8(row, col))

        for candidate in cells:
            for furniture_id in furniture_id_set:
                if (
                    furniture_id not in seen
                    and f"furniture:{furniture_id}" in candidate.tags
                ):
                    seen.add(furniture_id)
                    nearby.append(furniture_id)

    return nearby


def _step_direction(start: GridCoord, end: GridCoord) -> GridCoord:
    row_delta = end[0] - start[0]
    col_delta = end[1] - start[1]
    return _sign(row_delta), _sign(col_delta)


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


__all__ = [
    "compute_path_length",
    "compute_turn_count",
    "expand_path_cells",
    "grid_line_cells",
    "path_crosses_room",
    "path_near_furniture",
]
