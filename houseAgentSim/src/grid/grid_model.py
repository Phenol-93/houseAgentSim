"""供路径搜索和行为模拟使用的网格级空间模型。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.grid.grid_cell import GridCell


@dataclass
class Grid:
    cells: list[list[GridCell]]
    grid_size: float
    width: int
    height: int
    origin: tuple[float, float]
    activity_point_map: dict[str, tuple[int, int]] = field(default_factory=dict)

    def get_cell(self, row: int, col: int) -> GridCell | None:
        """根据网格坐标返回单元格；越界时返回 ``None``。"""
        if not self.in_bounds(row, col):
            return None
        return self.cells[row][col]

    def in_bounds(self, row: int, col: int) -> bool:
        """判断网格坐标是否位于当前网格范围内。"""
        return 0 <= row < self.height and 0 <= col < self.width

    def is_walkable(self, row: int, col: int) -> bool:
        """判断某个单元格是否可作为路径搜索候选点。"""
        cell = self.get_cell(row, col)
        if cell is None:
            return False
        return cell.walkable and not cell.blocked

    def neighbors_8(self, row: int, col: int) -> list[GridCell]:
        """返回边界内的 8 邻接单元格。"""
        neighbors: list[GridCell] = []
        for delta_row in (-1, 0, 1):
            for delta_col in (-1, 0, 1):
                if delta_row == 0 and delta_col == 0:
                    continue
                next_row = row + delta_row
                next_col = col + delta_col
                cell = self.get_cell(next_row, next_col)
                if cell is not None:
                    neighbors.append(cell)
        return neighbors

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        """把真实空间坐标转换为 ``(row, col)`` 网格坐标。"""
        origin_x, origin_y = self.origin
        col = math.floor((x - origin_x) / self.grid_size)
        row = math.floor((y - origin_y) / self.grid_size)
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        """返回某个网格单元中心点的真实空间坐标。"""
        origin_x, origin_y = self.origin
        x = origin_x + (col + 0.5) * self.grid_size
        y = origin_y + (row + 0.5) * self.grid_size
        return x, y

    def set_activity_point(self, point_id: str, row: int, col: int) -> None:
        """保存活动点 id 对应的网格坐标。"""
        if not self.in_bounds(row, col):
            raise ValueError(
                f"Activity point '{point_id}' grid coordinate is out of bounds: "
                f"({row}, {col})"
            )
        self.activity_point_map[point_id] = (row, col)

    def get_activity_grid(self, point_id: str) -> tuple[int, int] | None:
        """返回活动点 id 对应的网格坐标。"""
        return self.activity_point_map.get(point_id)


__all__ = ["Grid"]
