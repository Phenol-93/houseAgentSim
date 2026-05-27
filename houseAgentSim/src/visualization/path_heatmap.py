"""基于路径日志绘制住宅网格路径热力图。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.grid.grid_model import Grid
from src.layout.schema import Layout
from src.pathfinding.path_utils import expand_path_cells
from src.visualization.matplotlib_style import (
    apply_chinese_font_to_axes,
    chinese_font_properties,
    configure_matplotlib_font,
)


GridCoord = tuple[int, int]

configure_matplotlib_font()


def plot_path_heatmap(
    grid: Grid,
    path_log: pd.DataFrame,
    layout: Layout | None = None,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """绘制路径经过频次热力图，并可保存为 PNG。"""
    configure_matplotlib_font()
    heatmap = _build_heatmap_array(grid, path_log)
    blocked_mask = _blocked_mask(grid)

    figure, axes = plt.subplots(figsize=_figure_size(grid))
    _draw_heatmap(axes, grid, heatmap, blocked_mask)
    if layout is not None:
        _draw_layout_overlays(axes, layout)

    axes.set_aspect("equal")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("路径热力图", fontproperties=chinese_font_properties())
    axes.invert_yaxis()
    apply_chinese_font_to_axes(axes)
    figure.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)

    plt.close(figure)
    return heatmap


def _build_heatmap_array(grid: Grid, path_log: pd.DataFrame) -> np.ndarray:
    heatmap = np.zeros((grid.height, grid.width), dtype=float)
    if path_log.empty or "path" not in path_log.columns:
        return heatmap

    for value in path_log["path"]:
        for row, col in _expand_in_bounds_path(grid, _parse_path(value)):
            if grid.in_bounds(row, col):
                heatmap[row, col] += 1
    return heatmap


def _expand_in_bounds_path(grid: Grid, path: list[GridCoord]) -> list[GridCoord]:
    expanded: list[GridCoord] = []
    current_chunk: list[GridCoord] = []

    for coord in path:
        if grid.in_bounds(coord[0], coord[1]):
            current_chunk.append(coord)
            continue
        expanded.extend(expand_path_cells(current_chunk))
        current_chunk = []

    expanded.extend(expand_path_cells(current_chunk))
    return expanded


def _blocked_mask(grid: Grid) -> np.ndarray:
    mask = np.zeros((grid.height, grid.width), dtype=bool)
    for row in range(grid.height):
        for col in range(grid.width):
            cell = grid.get_cell(row, col)
            mask[row, col] = cell is None or cell.blocked or not cell.walkable
    return mask


def _draw_heatmap(
    axes,
    grid: Grid,
    heatmap: np.ndarray,
    blocked_mask: np.ndarray,
) -> None:
    origin_x, origin_y = grid.origin
    extent = [
        origin_x,
        origin_x + grid.width * grid.grid_size,
        origin_y + grid.height * grid.grid_size,
        origin_y,
    ]

    axes.imshow(
        blocked_mask,
        extent=extent,
        cmap=_blocked_cmap(),
        interpolation="none",
        vmin=0,
        vmax=1,
        zorder=0,
    )

    masked_heatmap = np.ma.masked_where(blocked_mask | (heatmap <= 0), heatmap)
    image = axes.imshow(
        masked_heatmap,
        extent=extent,
        cmap="inferno",
        interpolation="nearest",
        zorder=1,
    )
    if heatmap.max() > 0:
        plt.colorbar(image, ax=axes, fraction=0.046, pad=0.04, label="path count")


def _draw_layout_overlays(axes, layout: Layout) -> None:
    for room in layout.rooms:
        _draw_polygon_outline(axes, room.polygon, color="#2f2f2f", linewidth=1.0)
    for furniture in layout.furniture:
        _draw_polygon_outline(
            axes,
            furniture.polygon,
            color="#4f6f8f",
            linewidth=0.9,
            linestyle="--",
        )


def _draw_polygon_outline(
    axes,
    points: list[tuple[float, float]],
    color: str,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    if not points:
        return
    closed = points + [points[0]]
    xs = [point[0] for point in closed]
    ys = [point[1] for point in closed]
    axes.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, zorder=2)


def _parse_path(value: Any) -> list[GridCoord]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    if not isinstance(value, list):
        return []

    path: list[GridCoord] = []
    for item in value:
        coord = _parse_grid_coord(item)
        if coord is not None:
            path.append(coord)
    return path


def _parse_grid_coord(value: Any) -> GridCoord | None:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    row, col = value
    if not isinstance(row, (int, float)) or not isinstance(col, (int, float)):
        return None
    return int(row), int(col)


def _blocked_cmap():
    cmap = matplotlib.colors.ListedColormap(["white", "#d0d0d0"])
    cmap.set_bad("white")
    return cmap


def _figure_size(grid: Grid) -> tuple[float, float]:
    width = max(4.0, min(12.0, grid.width * 0.45))
    height = max(4.0, min(12.0, grid.height * 0.45))
    return width, height


__all__ = ["plot_path_heatmap"]
