"""自写住宅行为模拟使用的住户智能体模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.behavior import Behavior


@dataclass
class ResidentAgent:
    agent_id: str
    name: str
    age: int
    role: str
    mobility: float
    privacy_need: float
    noise_sensitivity: float
    schedule: list[Behavior]
    current_point: str
    personality: str = ""
    habits: list[str] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)
    routine_notes: str = ""
    profile_extras: dict[str, Any] = field(default_factory=dict)
    current_activity: str = "idle"
    waiting_records: list[dict[str, Any]] = field(default_factory=list)
    conflict_records: list[dict[str, Any]] = field(default_factory=list)

    def get_activity_at_time(self, current_time: str) -> Behavior | None:
        """返回当前时间正在生效的最高优先级行为。"""
        current_minutes = _time_to_minutes(current_time)
        matching: list[Behavior] = []

        for behavior in self.schedule:
            start = _time_to_minutes(behavior.time)
            end = start + behavior.duration
            if start <= current_minutes < end:
                matching.append(behavior)

        if not matching:
            return None
        return max(matching, key=lambda behavior: behavior.priority)

    def start_activity(self, behavior: Behavior) -> None:
        """开始执行行为，并把当前位置更新为行为目标点。"""
        self.current_activity = behavior.activity
        self.current_point = behavior.target

    def finish_activity(self) -> None:
        """把当前行为标记为结束。"""
        self.current_activity = "idle"

    def move_to(self, target_point: str) -> None:
        """更新居民当前所在活动点。"""
        self.current_point = target_point

    def record_waiting(self, time, target, reason) -> None:
        """记录等待事件，供后续分析使用。"""
        self.waiting_records.append(
            {
                "time": time,
                "target": target,
                "reason": reason,
            }
        )

    def record_conflict(self, time, conflict_type, description) -> None:
        """记录冲突事件，供后续分析使用。"""
        self.conflict_records.append(
            {
                "time": time,
                "conflict_type": conflict_type,
                "description": description,
            }
        )


def _time_to_minutes(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError(f"Time must use HH:MM format, got '{value}'")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Time must use HH:MM format, got '{value}'") from exc
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"Time is out of range, got '{value}'")
    return hour * 60 + minute


__all__ = ["ResidentAgent"]
