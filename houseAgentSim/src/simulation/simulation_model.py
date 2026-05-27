"""住宅一天行为模拟的核心模型。

该模块不负责生成居民行为，而是执行已经给定的 schedule：每个时间步检查
是否有新行为开始，调用路径搜索移动到目标活动点，并记录路径、等待、
占用和冲突日志。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from src.agents.behavior import Behavior
from src.agents.resident_agent import ResidentAgent
from src.grid.grid_model import Grid
from src.layout.schema import Layout
from src.pathfinding.astar import PathResult, find_path
from src.simulation.scheduler import MINUTES_PER_DAY, SimulationScheduler


@dataclass
class HousingSimulation:
    """基于户型、网格和居民列表运行一天离散时间模拟。"""
    layout: Layout
    grid: Grid
    agents: list[ResidentAgent]
    time_step: int = 5
    path_log: list[dict[str, Any]] = field(default_factory=list)
    wait_log: list[dict[str, Any]] = field(default_factory=list)
    occupancy_log: list[dict[str, Any]] = field(default_factory=list)
    conflict_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.scheduler = SimulationScheduler(time_step=self.time_step)
        self._active_occupancy: dict[str, list[dict[str, Any]]] = {}
        # 活动点可以设置 capacity，用于表达餐桌、沙发、床边照护点等多人共享位置。
        self._activity_point_capacity = {
            point.id: max(1, getattr(point, "capacity", 1))
            for point in self.layout.activity_points
        }
        self._target_aliases = self._build_target_aliases()

    def run_day(self) -> dict[str, list[dict[str, Any]]]:
        """运行完整一天，并返回四类日志。"""
        for current_time in self.scheduler.iter_day():
            self.step(current_time)
        self.release_finished_activities("24:00")
        return self._logs()

    def step(self, current_time: str) -> None:
        """推进一个时间步：释放过期占用、触发新行为、记录当前位置。"""
        self.release_finished_activities(current_time)
        for agent in self.agents:
            behavior = self._get_behavior_starting_at_time(agent, current_time)
            if behavior is not None:
                self.execute_behavior(agent, behavior, current_time)
            self._record_occupancy_state(agent, current_time)

    def execute_behavior(
        self,
        agent: ResidentAgent,
        behavior: Behavior,
        current_time: str,
    ) -> None:
        """执行某个居民在当前时间触发的行为。"""
        start_grid = self.grid.get_activity_grid(agent.current_point)
        target_point = self._resolve_behavior_target(agent, behavior)
        effective_behavior = (
            behavior
            if target_point == behavior.target
            else replace(behavior, target=target_point)
        )
        target_grid = self.grid.get_activity_grid(target_point)

        if start_grid is None:
            self.record_conflict(
                current_time,
                agent.agent_id,
                "missing_current_point",
                f"current point '{agent.current_point}' is not mapped to grid.",
            )
            return
        if target_grid is None:
            self.record_conflict(
                current_time,
                agent.agent_id,
                "missing_target_point",
                f"target point '{behavior.target}' is not mapped to grid.",
            )
            return

        # 所有移动都走统一寻路接口；当前实现为 Theta*，可避免路径过度折线化。
        path_result = find_path(self.grid, start_grid, target_grid)
        if not path_result.found:
            self.record_conflict(
                current_time,
                agent.agent_id,
                "path_not_found",
                path_result.reason or "pathfinding failed.",
                behavior=effective_behavior,
                start=start_grid,
                goal=target_grid,
                requested_target=behavior.target,
            )
            return

        if self.is_target_available(target_point, current_time, agent.agent_id):
            self.record_path(
                agent,
                effective_behavior,
                current_time,
                start_grid,
                target_grid,
                path_result,
                requested_target=behavior.target,
            )
            agent.start_activity(effective_behavior)
            self.occupy_target(agent, target_point, effective_behavior, current_time)
        else:
            occupied_by = [
                occupancy["agent_id"]
                for occupancy in self._active_occupancy.get(target_point, [])
            ]
            if len(occupied_by) == 1:
                reason = f"target '{target_point}' is occupied by agent '{occupied_by[0]}'."
            else:
                reason = (
                    f"target '{target_point}' is occupied by agents "
                    f"'{', '.join(occupied_by)}'."
                )
            self.record_wait(current_time, agent.agent_id, target_point, reason)
            agent.record_waiting(current_time, target_point, reason)

    def is_target_available(
        self,
        target_point: str,
        current_time: str,
        agent_id: str | None = None,
    ) -> bool:
        """判断目标活动点在当前时刻是否还有容量。"""
        self.release_finished_activities(current_time)
        occupancies = self._active_occupancy.get(target_point, [])
        if agent_id is not None:
            occupancies = [
                occupancy
                for occupancy in occupancies
                if occupancy["agent_id"] != agent_id
            ]
        return len(occupancies) < self._target_capacity(target_point)

    def occupy_target(
        self,
        agent: ResidentAgent,
        target_point: str,
        behavior: Behavior,
        current_time: str,
    ) -> None:
        """记录目标点占用，并计算该行为的释放时间。"""
        start_minutes = self.scheduler.time_to_minutes(current_time)
        self._active_occupancy[target_point] = [
            occupancy
            for occupancy in self._active_occupancy.get(target_point, [])
            if occupancy["agent_id"] != agent.agent_id
        ]
        self._active_occupancy.setdefault(target_point, []).append(
            {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "target": target_point,
            "activity": behavior.activity,
            "start_time": current_time,
            "start_minutes": start_minutes,
            "end_minutes": start_minutes + behavior.duration,
            "duration": behavior.duration,
            }
        )

    def release_finished_activities(self, current_time: str) -> None:
        """释放已经结束的活动占用，避免后续居民被错误阻塞。"""
        current_minutes = self.scheduler.time_to_minutes(current_time)
        if current_minutes == MINUTES_PER_DAY:
            current_minutes = MINUTES_PER_DAY

        for target, occupancies in list(self._active_occupancy.items()):
            active_occupancies = []
            for occupancy in occupancies:
                if occupancy["end_minutes"] <= current_minutes:
                    agent = self._agent_by_id(occupancy["agent_id"])
                    if (
                        agent is not None
                        and agent.current_activity == occupancy["activity"]
                    ):
                        agent.finish_activity()
                else:
                    active_occupancies.append(occupancy)
            if active_occupancies:
                self._active_occupancy[target] = active_occupancies
            else:
                self._active_occupancy.pop(target, None)

    def record_path(
        self,
        agent: ResidentAgent,
        behavior: Behavior,
        current_time: str,
        start: tuple[int, int],
        goal: tuple[int, int],
        path_result: PathResult,
        requested_target: str | None = None,
    ) -> None:
        """记录一次成功寻路结果。"""
        self.path_log.append(
            {
                "time": current_time,
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "activity": behavior.activity,
                "from_point": agent.current_point,
                "target_point": behavior.target,
                "requested_target": requested_target or behavior.target,
                "start_grid": start,
                "goal_grid": goal,
                "path": path_result.path,
                "path_length": path_result.path_length,
                "total_cost": path_result.total_cost,
                "turn_count": path_result.turn_count,
            }
        )

    def record_wait(
        self,
        time: str,
        agent_id: str,
        target: str,
        reason: str,
    ) -> None:
        """记录一次目标点等待事件。"""
        self.wait_log.append(
            {
                "time": time,
                "agent_id": agent_id,
                "target": target,
                "reason": reason,
            }
        )

    def record_conflict(
        self,
        time: str,
        agent_id: str,
        conflict_type: str,
        description: str,
        **extra: Any,
    ) -> None:
        """记录一次模拟冲突事件。"""
        record = {
            "time": time,
            "agent_id": agent_id,
            "conflict_type": conflict_type,
            "description": description,
        }
        record.update(extra)
        self.conflict_log.append(record)

        agent = self._agent_by_id(agent_id)
        if agent is not None:
            agent.record_conflict(time, conflict_type, description)

    def _record_occupancy_state(
        self,
        agent: ResidentAgent,
        current_time: str,
    ) -> None:
        self.occupancy_log.append(
            {
                "time": current_time,
                "agent_id": agent.agent_id,
                "agent_name": agent.name,
                "point": agent.current_point,
                "activity": agent.current_activity,
            }
        )

    def _get_behavior_starting_at_time(
        self,
        agent: ResidentAgent,
        current_time: str,
    ) -> Behavior | None:
        behaviors = [
            behavior for behavior in agent.schedule if behavior.time == current_time
        ]
        if not behaviors:
            return None
        return max(behaviors, key=lambda behavior: behavior.priority)

    def _agent_by_id(self, agent_id: str) -> ResidentAgent | None:
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def _target_capacity(self, target_point: str) -> int:
        return self._activity_point_capacity.get(target_point, 1)

    def _resolve_activity_target(self, target: str) -> str:
        if self.grid.get_activity_grid(target) is not None:
            return target
        return self._target_aliases.get(target, target)

    def _resolve_behavior_target(
        self,
        agent: ResidentAgent,
        behavior: Behavior,
    ) -> str:
        target = self._resolve_activity_target(behavior.target)
        if not self._is_care_behavior(agent, behavior):
            return target

        care_target = self._care_assist_target(target, behavior)
        if care_target is not None:
            return care_target
        return target

    def _is_care_behavior(self, agent: ResidentAgent, behavior: Behavior) -> bool:
        role = agent.role.lower()
        activity = behavior.activity.lower()
        tags = {tag.lower() for tag in behavior.tags}
        return (
            "caregiver" in role
            or "care" in tags
            or "care_assist" in tags
            or "assist" in activity
            or "care" in activity
        )

    def _care_assist_target(
        self,
        target: str,
        behavior: Behavior,
    ) -> str | None:
        activity = behavior.activity.lower()
        if target.startswith("bathroom") or target == "bathroom":
            return self._existing_activity_point("bathroom_assist")

        if target in {"master_bed", "bedroom_master_bed", "night_call"}:
            return self._existing_activity_point("master_transfer") or (
                self._existing_activity_point("master_bedside")
            )

        if target == "master_wardrobe_use" and (
            "dress" in activity or "assist" in activity
        ):
            return self._existing_activity_point("master_transfer")

        if target in {"dining_seat", "dining_table"} and (
            "medicine" in activity or "medication" in activity or "drug" in activity
        ):
            return self._existing_activity_point("medicine_table")

        return None

    def _existing_activity_point(self, point_id: str) -> str | None:
        if self.grid.get_activity_grid(point_id) is not None:
            return point_id
        return None

    def _build_target_aliases(self) -> dict[str, str]:
        points_by_room: dict[str, list[str]] = {}
        point_types: dict[str, str] = {}
        for point in self.layout.activity_points:
            points_by_room.setdefault(point.room, []).append(point.id)
            point_types[point.id] = point.activity_type

        aliases: dict[str, str] = {}
        for room in self.layout.rooms:
            default_point = self._default_point_for_room(
                points_by_room.get(room.id, []),
                point_types,
            )
            if default_point is None:
                continue
            aliases[room.id] = default_point
            aliases[room.type] = default_point

        aliases.update(
            {
                "master_bedroom": aliases.get("bedroom_master", "master_bed"),
                "second_bedroom": aliases.get("bedroom_second", "second_bedroom"),
                "living_room": aliases.get("living", "sofa"),
                "dining_room": aliases.get("dining", "dining_seat"),
                "bathroom": aliases.get("bathroom", "bathroom_toilet"),
                "kitchen": aliases.get("kitchen", "kitchen_counter"),
                "balcony": aliases.get("balcony", "balcony_use"),
                "hallway": aliases.get("foyer", "entry"),
                "corridor": aliases.get("foyer", "entry"),
            }
        )
        return {
            alias: target
            for alias, target in aliases.items()
            if target and self.grid.get_activity_grid(target) is not None
        }

    def _default_point_for_room(
        self,
        point_ids: list[str],
        point_types: dict[str, str],
    ) -> str | None:
        if not point_ids:
            return None
        priority = [
            "rest",
            "sleep",
            "eat",
            "toilet",
            "wash",
            "cook",
            "balcony",
            "entry",
            "transition",
        ]
        for activity_type in priority:
            for point_id in point_ids:
                if point_types.get(point_id) == activity_type:
                    return point_id
        return point_ids[0]

    def _logs(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "path_log": self.path_log,
            "wait_log": self.wait_log,
            "occupancy_log": self.occupancy_log,
            "conflict_log": self.conflict_log,
        }


__all__ = ["HousingSimulation"]
