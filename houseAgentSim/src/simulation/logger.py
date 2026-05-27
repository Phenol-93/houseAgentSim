"""模拟日志的 CSV 读写工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


LOG_FILE_NAMES = {
    "path_log": "path_log.csv",
    "wait_log": "wait_log.csv",
    "occupancy_log": "occupancy_log.csv",
    "conflict_log": "conflict_log.csv",
}

BASE_FIELDS = {
    "path_log": [
        "time",
        "agent_id",
        "agent_name",
        "activity",
        "from_point",
        "target_point",
        "start_grid",
        "goal_grid",
        "path",
        "path_length",
        "total_cost",
        "turn_count",
    ],
    "wait_log": ["time", "agent_id", "target", "reason"],
    "occupancy_log": ["time", "agent_id", "agent_name", "point", "activity"],
    "conflict_log": ["time", "agent_id", "conflict_type", "description"],
}


def save_log_csv(log: list[dict], path: str | Path) -> None:
    """把单个模拟日志列表保存为 CSV。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = _columns_for_path(output_path)
    if log:
        data_frame = pd.DataFrame(log)
        if columns:
            data_frame = data_frame.reindex(
                columns=columns
                + [column for column in data_frame.columns if column not in columns]
            )
    else:
        data_frame = pd.DataFrame(columns=columns)

    data_frame.to_csv(output_path, index=False, encoding="utf-8")


def save_all_logs(simulation_result: dict, output_dir: str | Path) -> None:
    """把四类标准模拟日志保存到指定目录。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for log_name, file_name in LOG_FILE_NAMES.items():
        log = simulation_result.get(log_name, [])
        if not isinstance(log, list):
            raise ValueError(f"simulation_result['{log_name}'] must be a list.")
        save_log_csv(log, output_path / file_name)


def load_log_csv(path: str | Path) -> pd.DataFrame:
    """把模拟日志 CSV 读取为 pandas DataFrame。"""
    return pd.read_csv(Path(path))


def _columns_for_path(path: Path) -> list[str]:
    for log_name, file_name in LOG_FILE_NAMES.items():
        if path.name == file_name:
            return BASE_FIELDS[log_name]
    return []


__all__ = ["load_log_csv", "save_all_logs", "save_log_csv"]
