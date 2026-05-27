"""单个网格单元的数据模型。"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GridCell:
    row: int
    col: int
    center: tuple[float, float]
    walkable: bool
    room_id: str | None
    blocked: bool
    block_type: str | None
    cost: float
    tags: set[str] = field(default_factory=set)


__all__ = ["GridCell"]
