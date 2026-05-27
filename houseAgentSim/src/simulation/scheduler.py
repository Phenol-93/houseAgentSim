"""住户行为模拟使用的离散时间调度器。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


MINUTES_PER_DAY = 24 * 60


@dataclass
class SimulationScheduler:
    start_time: str = "00:00"
    end_time: str = "24:00"
    time_step: int = 5

    def __post_init__(self) -> None:
        if self.time_step <= 0:
            raise ValueError("time_step must be greater than 0.")
        start_minutes = self.time_to_minutes(self.start_time)
        end_minutes = self.time_to_minutes(self.end_time)
        if end_minutes < start_minutes:
            raise ValueError("end_time must be greater than or equal to start_time.")

    def time_to_minutes(self, time_str: str) -> int:
        """把 ``HH:MM`` 时间转换为从午夜开始计算的分钟数。"""
        parts = time_str.split(":")
        if len(parts) != 2:
            raise ValueError(f"Time must use HH:MM format, got '{time_str}'")

        try:
            hour = int(parts[0])
            minute = int(parts[1])
        except ValueError as exc:
            raise ValueError(f"Time must use HH:MM format, got '{time_str}'") from exc

        if hour == 24 and minute == 0:
            return MINUTES_PER_DAY
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError(f"Time is out of range, got '{time_str}'")
        return hour * 60 + minute

    def minutes_to_time(self, minutes: int) -> str:
        """把从午夜开始计算的分钟数转换为 ``HH:MM`` 时间。"""
        if not 0 <= minutes <= MINUTES_PER_DAY:
            raise ValueError(f"Minutes must be between 0 and 1440, got {minutes}")
        if minutes == MINUTES_PER_DAY:
            return "24:00"

        hour = minutes // 60
        minute = minutes % 60
        return f"{hour:02d}:{minute:02d}"

    def iter_day(self) -> Iterator[str]:
        """按 time_step 依次生成模拟时间点。"""
        start_minutes = self.time_to_minutes(self.start_time)
        end_minutes = self.time_to_minutes(self.end_time)

        current = start_minutes
        while current < end_minutes:
            yield self.minutes_to_time(current)
            current += self.time_step

    def is_behavior_active(
        self,
        start_time: str,
        duration: int,
        current_time: str,
    ) -> bool:
        """判断某个行为在当前时间是否仍处于持续状态。"""
        if duration <= 0:
            return False

        start = self.time_to_minutes(start_time)
        current = self.time_to_minutes(current_time)
        if start == MINUTES_PER_DAY:
            start = 0
        if current == MINUTES_PER_DAY:
            current = 0

        duration = min(duration, MINUTES_PER_DAY)
        elapsed = (current - start) % MINUTES_PER_DAY
        return elapsed < duration


__all__ = ["SimulationScheduler"]
