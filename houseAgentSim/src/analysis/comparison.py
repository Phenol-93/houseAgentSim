"""对比原户型与改造户型的指标字典。

当前 Streamlit 界面已不展示改造前后对比，但保留该模块，便于后续课程
迭代时重新启用批量方案评估。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LOWER_BETTER_HINTS = {
    "cost",
    "count",
    "detour",
    "distance",
    "exposure",
    "interruption",
    "length",
    "risk",
    "score",
    "turn",
    "wait",
    "waiting",
}
HIGHER_BETTER_HINTS = {"access", "capacity", "comfort", "efficiency", "privacy"}
VALID_DIRECTIONS = {"lower_better", "higher_better"}


def compare_metrics(
    original: dict,
    renovated: dict,
    metric_directions: dict,
) -> dict:
    """对比原始结果和改造结果中共有的数值指标。"""
    original_flat = _flatten_numeric_metrics(original)
    renovated_flat = _flatten_numeric_metrics(renovated)
    shared_metrics = sorted(set(original_flat) & set(renovated_flat))

    comparisons: dict[str, dict[str, Any]] = {}
    for metric_name in shared_metrics:
        original_value = original_flat[metric_name]
        renovated_value = renovated_flat[metric_name]
        absolute_change = renovated_value - original_value
        percent_change = _percent_change(original_value, absolute_change)
        direction = _direction_for_metric(metric_name, metric_directions)

        comparisons[metric_name] = {
            "original": original_value,
            "renovated": renovated_value,
            "absolute_change": absolute_change,
            "percent_change": percent_change,
            "improvement_direction": direction,
            "improved": _is_improved(absolute_change, direction),
        }

    return {
        "metrics": comparisons,
        "overall_improvement_summary": _overall_summary(comparisons),
    }


def save_comparison_report_data(comparison: dict, path: str | Path) -> None:
    """把指标对比结果保存为 JSON。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, ensure_ascii=False, indent=2)
        file.write("\n")


def compare_metric_files(
    original_path: str | Path,
    renovated_path: str | Path,
    metric_directions: dict,
    output_path: str | Path | None = None,
) -> dict:
    """读取两个指标 JSON，完成对比，并可选择保存输出。"""
    original = _load_json(original_path)
    renovated = _load_json(renovated_path)
    comparison = compare_metrics(original, renovated, metric_directions)
    if output_path is not None:
        save_comparison_report_data(comparison, output_path)
    return comparison


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Metrics JSON must contain an object: {path}")
    return data


def _flatten_numeric_metrics(data: dict, prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in data.items():
        metric_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            flattened[metric_name] = float(value)
        elif isinstance(value, dict):
            flattened.update(_flatten_numeric_metrics(value, metric_name))
    return flattened


def _percent_change(original_value: float, absolute_change: float) -> float | None:
    if original_value == 0:
        return 0.0 if absolute_change == 0 else None
    return (absolute_change / original_value) * 100


def _direction_for_metric(metric_name: str, metric_directions: dict) -> str:
    if metric_name in metric_directions:
        return _validate_direction(metric_directions[metric_name], metric_name)

    top_level_name = metric_name.split(".", 1)[0]
    if top_level_name in metric_directions:
        return _validate_direction(metric_directions[top_level_name], top_level_name)

    lowered = metric_name.lower()
    if any(hint in lowered for hint in HIGHER_BETTER_HINTS):
        return "higher_better"
    if any(hint in lowered for hint in LOWER_BETTER_HINTS):
        return "lower_better"
    return "lower_better"


def _validate_direction(direction: str, metric_name: str) -> str:
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"Invalid metric direction for '{metric_name}': {direction}. "
            "Use 'lower_better' or 'higher_better'."
        )
    return direction


def _is_improved(absolute_change: float, direction: str) -> bool:
    if direction == "lower_better":
        return absolute_change < 0
    return absolute_change > 0


def _overall_summary(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    improved_metrics = [
        metric_name
        for metric_name, comparison in comparisons.items()
        if comparison["improved"]
    ]
    worsened_metrics = [
        metric_name
        for metric_name, comparison in comparisons.items()
        if _is_worse(comparison)
    ]
    unchanged_metrics = [
        metric_name
        for metric_name, comparison in comparisons.items()
        if comparison["absolute_change"] == 0
    ]
    total = len(comparisons)
    improved_count = len(improved_metrics)

    return {
        "compared_metric_count": total,
        "improved_count": improved_count,
        "worsened_count": len(worsened_metrics),
        "unchanged_count": len(unchanged_metrics),
        "improvement_rate": improved_count / total if total else 0.0,
        "overall_improved": improved_count > len(worsened_metrics),
        "improved_metrics": improved_metrics,
        "worsened_metrics": worsened_metrics,
        "unchanged_metrics": unchanged_metrics,
    }


def _is_worse(comparison: dict[str, Any]) -> bool:
    direction = comparison["improvement_direction"]
    absolute_change = comparison["absolute_change"]
    if absolute_change == 0:
        return False
    if direction == "lower_better":
        return absolute_change > 0
    return absolute_change < 0


__all__ = [
    "compare_metric_files",
    "compare_metrics",
    "save_comparison_report_data",
]
