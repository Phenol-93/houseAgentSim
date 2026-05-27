"""模拟完成后的家庭成员视角 AI 分析。

该模块读取已生成的指标和日志，把每位成员的路径、等待、冲突和常用空间
整理成 prompt，再调用硅基流动生成中文 Markdown 报告。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
from pandas.errors import EmptyDataError

from src.agents.ai_schedule_planner import (
    DEFAULT_SILICONFLOW_BASE_URL,
    DEFAULT_SILICONFLOW_MODEL,
    _load_local_env,
    _response_text,
)
from src.agents.resident_agent import ResidentAgent
from src.layout.schema import Layout


DEFAULT_TOP_ROWS = 12


def build_resident_perspective_prompt(
    layout: Layout,
    agents: list[ResidentAgent],
    metrics: dict[str, Any],
    logs: Mapping[str, pd.DataFrame],
) -> str:
    """构造成员视角分析 prompt，强调基于数据但保留生活化表达。"""
    payload = {
        "layout_summary": _layout_summary(layout),
        "resident_profiles": [_agent_summary(agent) for agent in agents],
        "simulation_metrics": metrics,
        "per_resident_simulation_summary": _per_resident_summary(agents, logs),
        "representative_logs": _representative_logs(logs),
    }

    return f"""你是一名住宅行为模拟与适老化居住体验研究助手。

任务：
请基于输入的户型信息、家庭成员画像、模拟指标和日志，分别带入每一位家庭成员的第一人称视角，分析他们在当前户型中的居住体验。
请把语气调整为“建设性、温和、会肯定现有设计价值”的住户访谈式表达。不要只挑毛病；每位成员都要先说清楚当前户型中让自己满意、方便、安心、愿意保留的地方，再谈不便和改造愿望。正向评价与保留建议的篇幅应明显增加，约占每位成员分析内容的 1/2 左右。

必须遵守：
1. 只基于输入数据解释，不得编造不存在的房间、家具、功能点、冲突或指标。
2. 如果某个判断缺少证据，请明确写“目前数据不足，需要补充……”，不要强行下结论。
3. 不要生成平面图，不要替用户重新设计完整户型。
4. 可以提出空间优化方向，但要说明这些建议对应哪条路径、等待、冲突、夜间风险或成员习惯。
5. 输出为中文 Markdown，偏建筑设计研究表达，但要保留居民视角的生活感。
6. “多夸一夸”不等于编造优点：可以基于低等待、少冲突、常用功能点可达、房间功能清晰、照护行为有落点、休息区相对稳定等输入信息，转化为居民感受到的便利、安心和可保留价值。
7. 如果某位成员没有等待或冲突记录，可以把它表述为“当前模拟中我没有明显被卡住/被干扰”，但必须提醒这只是基于本次模拟结果。

请严格按以下结构输出：

# 家庭成员视角空间体验分析

## 1. 总体观察
- 用 5-7 条概括当前户型对家庭日常生活的整体体验，其中至少 3 条写现有优点或可保留价值，再写主要矛盾。

## 2. 成员逐一分析
对每个家庭成员分别输出：
### 成员姓名 / 角色
#### 我喜欢和满意的地方
- 用第一人称写 3-5 条，尽量具体到房间、功能点、活动路径、照护关系或日常习惯。
- 这部分要比“不方便的地方”更充分，体现当前户型对该成员生活的积极支持。
#### 我觉得方便、愿意保留的空间关系
- 说明哪些功能邻接、家具位置、活动点或生活路线值得保留，不要一上来就建议拆改。
#### 我觉得不方便的地方
- 用 1-3 条说明即可，避免把报告写成纯负面评价。
#### 对我有风险或干扰的地方
#### 我觉得冗余或利用率不高的功能
#### 我还想增加或改善的功能
#### 支撑证据
- 引用具体指标、日志、时间、目标点或冲突类型。
- 同时引用支持“优点”的证据和支持“问题”的证据；如果缺少正向证据，请写“目前正向证据不足，但可以从……初步推断”。

## 3. 家庭内部的共同矛盾
- 先总结家庭成员之间配合较好的地方，再分析多人共享卫生间、餐桌、客厅、卧室、阳台等功能点时的冲突或互相影响。

## 4. 保留优先级与改造优先级建议
- 先列出“建议保留 / 强化”的内容，至少 3 条。
- 按“高 / 中 / 低”列出建议。
- 每条建议说明服务于哪位成员，以及改善哪个模拟问题。
- 改造建议应尽量建立在现有优点之上，避免把已经运行良好的生活关系推倒重来。

## 5. 后续需要补充的数据
- 如果要让分析更可信，还需要补充哪些标注、行为、访谈或实测数据。

输入数据：
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def generate_resident_perspective_report(
    layout: Layout,
    agents: list[ResidentAgent],
    metrics_path: str | Path,
    log_dir: str | Path,
    output_report_path: str | Path,
    output_prompt_path: str | Path,
    model: str | None = None,
    temperature: float = 0.35,
) -> dict[str, Any]:
    """调用模型生成成员视角报告，并保存 prompt 与 Markdown 结果。"""
    metrics = _load_metrics(metrics_path)
    logs = _load_logs(log_dir)
    prompt = build_resident_perspective_prompt(layout, agents, metrics, logs)

    prompt_path = Path(output_prompt_path)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    report = _call_siliconflow(prompt, model=model, temperature=temperature)
    report_path = Path(output_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    return {
        "prompt_path": str(prompt_path),
        "report_path": str(report_path),
        "metrics": metrics,
    }


def _call_siliconflow(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.35,
) -> str:
    _load_local_env()
    api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SILICONFLOW_API_KEY is not set. Set it in .env before running "
            "resident-perspective AI analysis."
        )

    selected_model = (
        model
        or os.getenv("SILICONFLOW_MODEL")
        or os.getenv("OPENAI_MODEL")
        or DEFAULT_SILICONFLOW_MODEL
    )
    base_url = os.getenv("SILICONFLOW_BASE_URL") or DEFAULT_SILICONFLOW_BASE_URL

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required for resident-perspective AI analysis."
        ) from exc

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你只输出中文 Markdown 报告。所有判断必须来自用户提供的模拟数据，"
                    "数据不足时明确说明缺失信息。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        stream=False,
        extra_body={"enable_thinking": False},
    )
    return _response_text(response)


def _layout_summary(layout: Layout) -> dict[str, Any]:
    return {
        "layout_id": layout.layout_id,
        "layout_name": layout.layout_name,
        "unit": layout.unit,
        "grid_size": layout.grid_size,
        "rooms": [
            {
                "id": room.id,
                "name": room.name,
                "type": room.type,
                "capacity": room.capacity,
            }
            for room in layout.rooms
        ],
        "activity_points": [
            {
                "id": point.id,
                "name": point.name,
                "room": point.room,
                "activity_type": point.activity_type,
            }
            for point in layout.activity_points
        ],
        "furniture": [
            {
                "id": item.id,
                "name": item.name,
                "room": item.room,
                "walkable": item.walkable,
            }
            for item in layout.furniture
        ],
        "constraints": layout.constraints,
    }


def _agent_summary(agent: ResidentAgent) -> dict[str, Any]:
    return {
        "agent_id": agent.agent_id,
        "name": agent.name,
        "age": agent.age,
        "role": agent.role,
        "mobility": agent.mobility,
        "privacy_need": agent.privacy_need,
        "noise_sensitivity": agent.noise_sensitivity,
        "current_point": agent.current_point,
        "personality": agent.personality,
        "habits": agent.habits,
        "needs": agent.needs,
        "routine_notes": agent.routine_notes,
    }


def _per_resident_summary(
    agents: list[ResidentAgent],
    logs: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    """按居民聚合日志，减少 prompt 中无关原始记录的噪声。"""
    path_log = logs.get("path_log", pd.DataFrame())
    wait_log = logs.get("wait_log", pd.DataFrame())
    conflict_log = logs.get("conflict_log", pd.DataFrame())
    occupancy_log = logs.get("occupancy_log", pd.DataFrame())

    summaries: dict[str, dict[str, Any]] = {}
    for agent in agents:
        agent_id = agent.agent_id
        agent_paths = _filter_agent(path_log, agent_id)
        agent_waits = _filter_agent(wait_log, agent_id)
        agent_conflicts = _filter_agent(conflict_log, agent_id)
        agent_occupancy = _filter_agent(occupancy_log, agent_id)

        summaries[agent_id] = {
            "name": agent.name,
            "total_path_length": _numeric_sum(agent_paths, "path_length"),
            "avg_path_length": _numeric_mean(agent_paths, "path_length"),
            "avg_turn_count": _numeric_mean(agent_paths, "turn_count"),
            "path_event_count": int(len(agent_paths)),
            "wait_event_count": int(len(agent_waits)),
            "conflict_event_count": int(len(agent_conflicts)),
            "frequent_targets": _top_values(agent_paths, "target_point"),
            "frequent_occupied_points": _top_values(agent_occupancy, "current_point"),
            "wait_targets": _top_values(agent_waits, "target"),
            "conflict_types": _top_values(agent_conflicts, "conflict_type"),
            "representative_waits": _records(agent_waits, limit=DEFAULT_TOP_ROWS),
            "representative_conflicts": _records(
                agent_conflicts, limit=DEFAULT_TOP_ROWS
            ),
            "longest_paths": _longest_paths(agent_paths),
        }
    return summaries


def _representative_logs(logs: Mapping[str, pd.DataFrame]) -> dict[str, list[dict]]:
    return {
        "path_log_longest": _longest_paths(logs.get("path_log", pd.DataFrame())),
        "wait_log": _records(logs.get("wait_log", pd.DataFrame()), DEFAULT_TOP_ROWS),
        "conflict_log": _records(
            logs.get("conflict_log", pd.DataFrame()), DEFAULT_TOP_ROWS
        ),
    }


def _load_metrics(path: str | Path) -> dict[str, Any]:
    metrics_path = Path(path)
    if not metrics_path.exists() or metrics_path.stat().st_size == 0:
        return {}
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_logs(log_dir: str | Path) -> dict[str, pd.DataFrame]:
    directory = Path(log_dir)
    return {
        "path_log": _read_csv(directory / "path_log.csv"),
        "wait_log": _read_csv(directory / "wait_log.csv"),
        "occupancy_log": _read_csv(directory / "occupancy_log.csv"),
        "conflict_log": _read_csv(directory / "conflict_log.csv"),
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _filter_agent(data_frame: pd.DataFrame, agent_id: str) -> pd.DataFrame:
    if data_frame.empty or "agent_id" not in data_frame.columns:
        return pd.DataFrame()
    return data_frame[data_frame["agent_id"].astype(str) == agent_id]


def _numeric_sum(data_frame: pd.DataFrame, column: str) -> float:
    if data_frame.empty or column not in data_frame.columns:
        return 0.0
    return float(pd.to_numeric(data_frame[column], errors="coerce").fillna(0).sum())


def _numeric_mean(data_frame: pd.DataFrame, column: str) -> float | None:
    if data_frame.empty or column not in data_frame.columns:
        return None
    values = pd.to_numeric(data_frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _top_values(data_frame: pd.DataFrame, column: str, limit: int = 6) -> dict[str, int]:
    if data_frame.empty or column not in data_frame.columns:
        return {}
    counts = data_frame[column].fillna("").astype(str).value_counts().head(limit)
    return {key: int(value) for key, value in counts.items() if key}


def _records(data_frame: pd.DataFrame, limit: int) -> list[dict]:
    if data_frame.empty:
        return []
    records = data_frame.head(limit).fillna("").to_dict(orient="records")
    return [_json_safe_record(record) for record in records]


def _longest_paths(data_frame: pd.DataFrame, limit: int = 8) -> list[dict]:
    if data_frame.empty or "path_length" not in data_frame.columns:
        return []
    data = data_frame.copy()
    data["_path_length_sort"] = pd.to_numeric(data["path_length"], errors="coerce")
    data = data.sort_values("_path_length_sort", ascending=False).head(limit)
    columns = [
        column
        for column in (
            "time",
            "agent_id",
            "agent_name",
            "activity",
            "from_point",
            "target_point",
            "path_length",
            "turn_count",
        )
        if column in data.columns
    ]
    return [_json_safe_record(record) for record in data[columns].fillna("").to_dict("records")]


def _json_safe_record(record: Mapping[str, Any]) -> dict[str, Any]:
    safe_record: dict[str, Any] = {}
    for key, value in record.items():
        if hasattr(value, "item"):
            value = value.item()
        safe_record[str(key)] = value
    return safe_record


__all__ = [
    "build_resident_perspective_prompt",
    "generate_resident_perspective_report",
]
