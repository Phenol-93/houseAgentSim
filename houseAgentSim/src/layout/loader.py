"""手动标注住宅户型 JSON 的读取与保存工具。"""

from __future__ import annotations

import json
from dataclasses import asdict
from json import JSONDecodeError
from pathlib import Path
from typing import Any

from src.layout.schema import Layout


def load_layout(path: str | Path) -> Layout:
    """读取户型 JSON 文件，并转换为 ``Layout`` 对象。"""
    layout_path = Path(path)
    if not layout_path.exists():
        raise FileNotFoundError(f"Layout file does not exist: {layout_path}")
    if not layout_path.is_file():
        raise ValueError(f"Layout path is not a file: {layout_path}")

    try:
        with layout_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in layout file {layout_path}: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Layout JSON root must be an object, got {type(data).__name__}"
        )

    return Layout.from_dict(_normalize_layout_dict(data))


def save_layout(layout: Layout, path: str | Path) -> None:
    """把 ``Layout`` 对象保存为 JSON，便于后续编辑或复用。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(layout), file, ensure_ascii=False, indent=2)
        file.write("\n")


def _normalize_layout_dict(data: dict[str, Any]) -> dict[str, Any]:
    """兼容旧字段名，例如把门洞字段 ``from`` 转为 ``from_room``。"""
    normalized = dict(data)
    doors = normalized.get("doors")

    if isinstance(doors, list):
        normalized["doors"] = [
            _normalize_door_dict(door) if isinstance(door, dict) else door
            for door in doors
        ]

    return normalized


def _normalize_door_dict(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    if "from_room" not in normalized and "from" in normalized:
        normalized["from_room"] = normalized["from"]
    return normalized


__all__ = ["load_layout", "save_layout"]
