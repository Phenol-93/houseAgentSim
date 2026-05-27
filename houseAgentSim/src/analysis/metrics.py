"""从模拟 CSV 日志中汇总空间行为指标。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError


DEFAULT_WAIT_MINUTES = 5
BATHROOM_KEYWORDS = {
    "bath",
    "bathroom",
    "toilet",
    "wc",
    "washroom",
    "restroom",
    "卫生",
    "厕所",
    "洗手",
    "浴",
}
ELDERLY_KEYWORDS = {"elder", "elderly", "senior", "grandma", "grandpa", "老人", "长者"}
MORNING_PEAK_START = 6 * 60
MORNING_PEAK_END = 9 * 60
NIGHT_START = 22 * 60
NIGHT_END = 6 * 60


def compute_metrics(log_dir: str | Path) -> dict:
    """读取标准日志目录并计算高层模拟指标。"""
    log_path = Path(log_dir)
    path_log = _read_log(log_path / "path_log.csv")
    wait_log = _read_log(log_path / "wait_log.csv")
    occupancy_log = _read_log(log_path / "occupancy_log.csv")
    conflict_log = _read_log(log_path / "conflict_log.csv")

    _ = occupancy_log

    metrics = {
        "total_path_length_by_agent": _total_path_length_by_agent(path_log),
        "total_path_length_all": _total_path_length_all(path_log),
        "elderly_night_path_length_avg": _elderly_night_average(
            path_log, conflict_log, "path_length"
        ),
        "elderly_night_turn_count_avg": _elderly_night_average(
            path_log, conflict_log, "turn_count"
        ),
        "bathroom_waiting_total": _bathroom_waiting_total(wait_log),
        "bathroom_peak_waiting_avg": _bathroom_peak_waiting_avg(wait_log),
        "work_area_interruption_count": _conflict_type_count(
            conflict_log, "work_interference"
        ),
        "visitor_privacy_exposure_score": _privacy_exposure_score(conflict_log),
        "furniture_detour_distance": _furniture_detour_distance(
            path_log, conflict_log
        ),
        "conflict_count_by_type": _conflict_count_by_type(conflict_log),
    }
    return metrics


def save_metrics(metrics: dict, path: str | Path) -> None:
    """把指标字典保存为 JSON 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _total_path_length_by_agent(path_log: pd.DataFrame) -> dict[str, float]:
    if path_log.empty or not {"agent_id", "path_length"}.issubset(path_log.columns):
        return {}
    data = path_log.copy()
    data["path_length"] = _to_numeric(data["path_length"])
    grouped = data.groupby("agent_id")["path_length"].sum()
    return {str(agent_id): float(value) for agent_id, value in grouped.items()}


def _total_path_length_all(path_log: pd.DataFrame) -> float:
    if path_log.empty or "path_length" not in path_log.columns:
        return 0.0
    return float(_to_numeric(path_log["path_length"]).sum())


def _elderly_night_average(
    path_log: pd.DataFrame,
    conflict_log: pd.DataFrame,
    metric_column: str,
) -> float:
    conflict_values = _values_from_conflict_events(
        conflict_log, "elderly_night_risk", metric_column
    )
    if conflict_values:
        return float(sum(conflict_values) / len(conflict_values))

    if path_log.empty or metric_column not in path_log.columns:
        return 0.0

    records = []
    for _, record in path_log.iterrows():
        if not _is_night_time(str(record.get("time", ""))):
            continue
        text = " ".join(
            str(record.get(field, ""))
            for field in ("agent_id", "agent_name", "activity", "target_point")
        )
        if _contains_keyword(text, ELDERLY_KEYWORDS) and _contains_keyword(
            text, BATHROOM_KEYWORDS
        ):
            records.append(record)

    if not records:
        return 0.0
    values = _to_numeric(pd.Series([record.get(metric_column) for record in records]))
    return float(values.mean()) if not values.empty else 0.0


def _bathroom_waiting_total(wait_log: pd.DataFrame) -> float:
    bathroom_waits = _bathroom_wait_records(wait_log)
    if bathroom_waits.empty:
        return 0.0
    return float(_wait_minutes(bathroom_waits).sum())


def _bathroom_peak_waiting_avg(wait_log: pd.DataFrame) -> float:
    bathroom_waits = _bathroom_wait_records(wait_log)
    if bathroom_waits.empty or "time" not in bathroom_waits.columns:
        return 0.0

    peak_waits = bathroom_waits[
        bathroom_waits["time"].apply(lambda value: _is_morning_peak(str(value)))
    ]
    if peak_waits.empty:
        return 0.0
    return float(_wait_minutes(peak_waits).mean())


def _bathroom_wait_records(wait_log: pd.DataFrame) -> pd.DataFrame:
    if wait_log.empty:
        return wait_log
    columns = [column for column in ("target", "reason", "activity") if column in wait_log]
    if not columns:
        return wait_log.iloc[0:0]
    mask = wait_log[columns].fillna("").agg(" ".join, axis=1).apply(
        lambda value: _contains_keyword(value, BATHROOM_KEYWORDS)
    )
    return wait_log[mask]


def _wait_minutes(wait_log: pd.DataFrame) -> pd.Series:
    for column in ("wait_minutes", "waiting_minutes", "duration", "duration_minutes"):
        if column in wait_log.columns:
            return _to_numeric(wait_log[column]).fillna(0)
    return pd.Series([DEFAULT_WAIT_MINUTES] * len(wait_log), index=wait_log.index)


def _conflict_type_count(conflict_log: pd.DataFrame, conflict_type: str) -> int:
    if conflict_log.empty or "conflict_type" not in conflict_log.columns:
        return 0
    return int((conflict_log["conflict_type"] == conflict_type).sum())


def _privacy_exposure_score(conflict_log: pd.DataFrame) -> float:
    if conflict_log.empty or "conflict_type" not in conflict_log.columns:
        return 0.0
    privacy_events = conflict_log[conflict_log["conflict_type"] == "privacy_exposure"]
    if privacy_events.empty:
        return 0.0
    for column in ("score", "privacy_score", "exposure_score", "severity"):
        if column in privacy_events.columns:
            return float(_to_numeric(privacy_events[column]).fillna(0).sum())
    return float(len(privacy_events))


def _furniture_detour_distance(
    path_log: pd.DataFrame,
    conflict_log: pd.DataFrame,
) -> float:
    for data in (path_log, conflict_log):
        if data.empty:
            continue
        for column in ("detour_distance", "furniture_detour_distance", "extra_distance"):
            if column in data.columns:
                return float(_to_numeric(data[column]).fillna(0).sum())

    if conflict_log.empty or "conflict_type" not in conflict_log.columns:
        return 0.0
    detours = conflict_log[conflict_log["conflict_type"] == "furniture_detour"]
    if detours.empty:
        return 0.0
    return float(len(detours) * DEFAULT_WAIT_MINUTES)


def _conflict_count_by_type(conflict_log: pd.DataFrame) -> dict[str, int]:
    if conflict_log.empty or "conflict_type" not in conflict_log.columns:
        return {}
    counts = conflict_log["conflict_type"].fillna("unknown").value_counts()
    return {str(conflict_type): int(count) for conflict_type, count in counts.items()}


def _values_from_conflict_events(
    conflict_log: pd.DataFrame,
    conflict_type: str,
    column: str,
) -> list[float]:
    if (
        conflict_log.empty
        or "conflict_type" not in conflict_log.columns
        or column not in conflict_log.columns
    ):
        return []
    events = conflict_log[conflict_log["conflict_type"] == conflict_type]
    values = _to_numeric(events[column]).dropna()
    return [float(value) for value in values]


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


def _is_morning_peak(time: str) -> bool:
    minutes = _time_to_minutes_or_none(time)
    if minutes is None:
        return False
    return MORNING_PEAK_START <= minutes < MORNING_PEAK_END


def _is_night_time(time: str) -> bool:
    minutes = _time_to_minutes_or_none(time)
    if minutes is None:
        return False
    return minutes >= NIGHT_START or minutes < NIGHT_END


def _time_to_minutes_or_none(time: str) -> int | None:
    parts = time.split(":")
    if len(parts) != 2:
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None
    if hour == 24 and minute == 0:
        return 24 * 60
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None
    return hour * 60 + minute


def _contains_keyword(text: str, keywords: set[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


__all__ = ["compute_metrics", "save_metrics"]
