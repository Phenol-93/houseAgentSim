"""网格模型上的 Theta* 路径搜索。

The module keeps the historical ``astar`` name so existing imports continue to
work, but ``find_path`` now runs Theta* and can return any-angle shortcuts when
there is clear line of sight across walkable cells.

文件名仍叫 ``astar.py`` 是为了兼容早期导入路径；实际算法已经换成 Theta*，
在有视线可达时会把路径拉直，更接近人在室内空间中的自然行走方式。
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from src.grid.grid_cell import GridCell
from src.grid.grid_model import Grid
from src.pathfinding.path_utils import (
    compute_path_length,
    compute_turn_count,
    grid_line_cells,
)


GridCoord = tuple[int, int]


@dataclass
class PathResult:
    found: bool
    path: list[GridCoord]
    path_length: float
    total_cost: float
    turn_count: int
    reason: str | None


def find_path(
    grid: Grid,
    start: GridCoord,
    goal: GridCoord,
    allow_diagonal: bool = True,
) -> PathResult:
    """在两个网格点之间搜索路径，并返回长度、成本和转折次数。"""
    start_problem = _coordinate_problem(grid, start, "start")
    if start_problem is not None:
        return _not_found(start_problem)

    goal_problem = _coordinate_problem(grid, goal, "goal")
    if goal_problem is not None:
        return _not_found(goal_problem)

    if start == goal:
        return PathResult(
            found=True,
            path=[start],
            path_length=0.0,
            total_cost=0.0,
            turn_count=0,
            reason=None,
        )

    # open_heap 按 f = g + h 排序；counter 用于避免优先级相同时比较坐标。
    open_heap: list[tuple[float, float, int, GridCoord]] = []
    counter = 0
    heapq.heappush(open_heap, (_heuristic(start, goal, grid.grid_size), 0.0, counter, start))

    came_from: dict[GridCoord, GridCoord] = {start: start}
    cost_so_far: dict[GridCoord, float] = {start: 0.0}
    closed: set[GridCoord] = set()

    while open_heap:
        _, current_cost, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = _reconstruct_path(came_from, start, goal)
            return PathResult(
                found=True,
                path=path,
                path_length=compute_path_length(path, grid.grid_size),
                total_cost=current_cost,
                turn_count=compute_turn_count(path),
                reason=None,
            )

        closed.add(current)
        for neighbor in _neighbors(grid, current, allow_diagonal):
            neighbor_coord = (neighbor.row, neighbor.col)
            if neighbor_coord in closed:
                continue

            parent = came_from.get(current, current)
            # Theta* 的关键：如果当前节点的父节点能直视邻居，就跳过当前节点，
            # 直接连接父节点和邻居，从而减少不必要的折线。
            if (
                allow_diagonal
                and parent != current
                and _line_of_sight(grid, parent, neighbor_coord)
            ):
                new_cost = cost_so_far[parent] + _segment_cost(
                    grid, parent, neighbor_coord
                )
                new_parent = parent
            else:
                new_cost = cost_so_far[current] + _movement_cost(
                    grid, current, neighbor_coord, neighbor
                )
                new_parent = current

            if new_cost >= cost_so_far.get(neighbor_coord, math.inf):
                continue

            cost_so_far[neighbor_coord] = new_cost
            came_from[neighbor_coord] = new_parent
            counter += 1
            priority = new_cost + _heuristic(neighbor_coord, goal, grid.grid_size)
            heapq.heappush(open_heap, (priority, new_cost, counter, neighbor_coord))

    return _not_found(f"No path found from {start} to {goal}.")


def _coordinate_problem(grid: Grid, coord: GridCoord, label: str) -> str | None:
    row, col = coord
    if not grid.in_bounds(row, col):
        return f"{label} {coord} is out of bounds."
    if not grid.is_walkable(row, col):
        return f"{label} {coord} is not walkable."
    return None


def _neighbors(grid: Grid, coord: GridCoord, allow_diagonal: bool) -> list[GridCell]:
    """返回可通行邻居；斜向移动时阻止从墙角或家具角穿过去。"""
    row, col = coord
    neighbors: list[GridCell] = []
    for candidate in grid.neighbors_8(row, col):
        row_delta = candidate.row - row
        col_delta = candidate.col - col
        is_diagonal = row_delta != 0 and col_delta != 0
        if is_diagonal and not allow_diagonal:
            continue
        if not grid.is_walkable(candidate.row, candidate.col):
            continue
        if is_diagonal and _cuts_blocked_corner(grid, row, col, row_delta, col_delta):
            continue
        neighbors.append(candidate)
    return neighbors


def _cuts_blocked_corner(
    grid: Grid,
    row: int,
    col: int,
    row_delta: int,
    col_delta: int,
) -> bool:
    return not (
        grid.is_walkable(row + row_delta, col)
        and grid.is_walkable(row, col + col_delta)
    )


def _movement_cost(
    grid: Grid,
    current: GridCoord,
    neighbor: GridCoord,
    neighbor_cell: GridCell,
) -> float:
    row_delta = abs(neighbor[0] - current[0])
    col_delta = abs(neighbor[1] - current[1])
    distance_cost = math.hypot(row_delta, col_delta) * grid.grid_size
    return distance_cost * neighbor_cell.cost


def _segment_cost(grid: Grid, start: GridCoord, end: GridCoord) -> float:
    traversed = grid_line_cells(start, end)
    if len(traversed) <= 1:
        return 0.0

    cost_values: list[float] = []
    for current in traversed[1:]:
        cell = grid.get_cell(current[0], current[1])
        if cell is None:
            return math.inf
        cost_values.append(cell.cost)
    average_cost = sum(cost_values) / len(cost_values)
    return _heuristic(start, end, grid.grid_size) * average_cost


def _line_of_sight(grid: Grid, start: GridCoord, end: GridCoord) -> bool:
    """判断两个网格点之间是否存在无遮挡直线路径。"""
    cells = grid_line_cells(start, end)
    if not cells:
        return False

    previous = cells[0]
    if not grid.is_walkable(previous[0], previous[1]):
        return False

    for current in cells[1:]:
        if not grid.is_walkable(current[0], current[1]):
            return False

        row_delta = current[0] - previous[0]
        col_delta = current[1] - previous[1]
        if row_delta != 0 and col_delta != 0:
            row_step = 1 if row_delta > 0 else -1
            col_step = 1 if col_delta > 0 else -1
            if _cuts_blocked_corner(
                grid,
                previous[0],
                previous[1],
                row_step,
                col_step,
            ):
                return False
        previous = current
    return True


def _heuristic(coord: GridCoord, goal: GridCoord, grid_size: float) -> float:
    return math.hypot(goal[0] - coord[0], goal[1] - coord[1]) * grid_size


def _reconstruct_path(
    came_from: dict[GridCoord, GridCoord],
    start: GridCoord,
    goal: GridCoord,
) -> list[GridCoord]:
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _not_found(reason: str) -> PathResult:
    return PathResult(
        found=False,
        path=[],
        path_length=0.0,
        total_cost=0.0,
        turn_count=0,
        reason=reason,
    )


__all__ = ["PathResult", "find_path"]
