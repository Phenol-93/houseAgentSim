"""在住宅户型平面上绘制冲突事件点位图。"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon

from src.layout.schema import Layout
from src.visualization.matplotlib_style import (
    apply_chinese_font_to_axes,
    chinese_font_properties,
    configure_matplotlib_font,
)


CONFLICT_STYLES = {
    "bathroom_queue": {"marker": "o", "color": "#d62728"},
    "work_interference": {"marker": "X", "color": "#ff7f0e"},
    "privacy_exposure": {"marker": "^", "color": "#9467bd"},
    "elderly_night_risk": {"marker": "s", "color": "#1f77b4"},
    "furniture_detour": {"marker": "D", "color": "#2ca02c"},
}
DEFAULT_STYLE = {"marker": "o", "color": "#111111"}

configure_matplotlib_font()


def plot_conflict_map(
    layout: Layout,
    conflict_log: pd.DataFrame,
    output_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """把冲突事件绘制到户型图上，并可保存为 PNG。"""
    configure_matplotlib_font()
    font_properties = chinese_font_properties()
    figure, axes = plt.subplots(figsize=_figure_size(layout))

    _draw_rooms(axes, layout)
    _draw_furniture(axes, layout)
    _draw_activity_points(axes, layout)
    plotted_events = _draw_conflicts(axes, layout, conflict_log)

    axes.set_aspect("equal")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title("冲突点图", fontproperties=font_properties)
    axes.invert_yaxis()
    if plotted_events:
        axes.legend(loc="best", fontsize=8)
    apply_chinese_font_to_axes(axes)
    figure.tight_layout()

    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=160)

    plt.close(figure)
    return plotted_events


def _draw_rooms(axes, layout: Layout) -> None:
    for room in layout.rooms:
        _draw_polygon_fill(
            axes,
            room.polygon,
            edge_color="#2f2f2f",
            face_color="#f8f8f8",
            linewidth=1.1,
            zorder=0,
        )
        center = _polygon_center(room.polygon)
        axes.text(
            center[0],
            center[1],
            room.name,
            fontsize=8,
            ha="center",
            va="center",
            color="#4d4d4d",
            fontproperties=chinese_font_properties(),
            zorder=1,
        )


def _draw_furniture(axes, layout: Layout) -> None:
    for furniture in layout.furniture:
        _draw_polygon_fill(
            axes,
            furniture.polygon,
            edge_color="#607d8b",
            face_color="#d9e5ea",
            linewidth=0.8,
            linestyle="--",
            zorder=2,
        )


def _draw_activity_points(axes, layout: Layout) -> None:
    if not layout.activity_points:
        return
    xs = [point.position[0] for point in layout.activity_points]
    ys = [point.position[1] for point in layout.activity_points]
    axes.scatter(
        xs,
        ys,
        marker=".",
        s=35,
        color="#333333",
        alpha=0.7,
        label="activity point",
        zorder=3,
    )


def _draw_conflicts(
    axes,
    layout: Layout,
    conflict_log: pd.DataFrame,
) -> list[dict[str, Any]]:
    plotted_events: list[dict[str, Any]] = []
    if conflict_log.empty:
        return plotted_events

    labeled_types: set[str] = set()
    for index, record in conflict_log.iterrows():
        conflict_type = str(record.get("conflict_type", "unknown"))
        position = _conflict_position(layout, record)
        if position is None:
            continue

        style = CONFLICT_STYLES.get(conflict_type, DEFAULT_STYLE)
        label = conflict_type if conflict_type not in labeled_types else None
        labeled_types.add(conflict_type)

        axes.scatter(
            [position[0]],
            [position[1]],
            marker=style["marker"],
            s=90,
            color=style["color"],
            edgecolors="white",
            linewidths=0.8,
            label=label,
            zorder=5,
        )
        axes.text(
            position[0] + 0.08,
            position[1] + 0.08,
            _event_label(index, conflict_type),
            fontsize=7,
            color=style["color"],
            fontproperties=chinese_font_properties(),
            zorder=6,
        )

        plotted_events.append(
            {
                "index": int(index),
                "conflict_type": conflict_type,
                "position": position,
            }
        )

    return plotted_events


def _conflict_position(layout: Layout, record) -> tuple[float, float] | None:
    activity_points = {point.id: point.position for point in layout.activity_points}

    for field in ("target", "target_point", "work_point", "point", "activity_point"):
        value = record.get(field)
        if value in activity_points:
            return activity_points[value]

    coord = _as_coordinate(record.get("position"))
    if coord is not None:
        return coord

    room_id = _room_id_from_record(record)
    if room_id is not None:
        room = next((room for room in layout.rooms if room.id == room_id), None)
        if room is not None:
            return _polygon_center(room.polygon)

    return _layout_center(layout)


def _room_id_from_record(record) -> str | None:
    for field in ("room", "room_id"):
        value = record.get(field)
        if isinstance(value, str) and value:
            return value

    private_rooms = record.get("private_rooms")
    if isinstance(private_rooms, str):
        try:
            private_rooms = ast.literal_eval(private_rooms)
        except (SyntaxError, ValueError):
            private_rooms = [private_rooms]
    if isinstance(private_rooms, list) and private_rooms:
        return str(private_rooms[0])
    return None


def _as_coordinate(value: Any) -> tuple[float, float] | None:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x, y = value
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return float(x), float(y)


def _draw_polygon_fill(
    axes,
    points: list[tuple[float, float]],
    edge_color: str,
    face_color: str,
    linewidth: float,
    linestyle: str = "-",
    zorder: int = 1,
) -> None:
    if not points:
        return
    polygon = plt.Polygon(
        points,
        closed=True,
        edgecolor=edge_color,
        facecolor=face_color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=zorder,
    )
    axes.add_patch(polygon)


def _polygon_center(points: list[tuple[float, float]]) -> tuple[float, float]:
    polygon = Polygon(points)
    centroid = polygon.centroid
    return float(centroid.x), float(centroid.y)


def _layout_center(layout: Layout) -> tuple[float, float] | None:
    if not layout.boundary:
        return None
    return _polygon_center(layout.boundary)


def _event_label(index, conflict_type: str) -> str:
    short_type = {
        "bathroom_queue": "bath",
        "work_interference": "work",
        "privacy_exposure": "privacy",
        "elderly_night_risk": "elder",
        "furniture_detour": "furniture",
    }.get(conflict_type, conflict_type)
    return f"{index}: {short_type}"


def _figure_size(layout: Layout) -> tuple[float, float]:
    if not layout.boundary:
        return 7.0, 6.0
    xs = [point[0] for point in layout.boundary]
    ys = [point[1] for point in layout.boundary]
    width = max(5.0, min(12.0, (max(xs) - min(xs)) * 1.2))
    height = max(4.0, min(12.0, (max(ys) - min(ys)) * 1.2))
    return width, height


__all__ = ["plot_conflict_map"]
