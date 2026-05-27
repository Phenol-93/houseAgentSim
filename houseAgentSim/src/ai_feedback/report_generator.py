"""本地诊断报告生成脚手架。

当前 Streamlit 页面主要使用家庭成员视角分析，本模块作为旧版诊断报告
流程保留，便于后续恢复“整体空间诊断”功能。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from src.ai_feedback.prompt_builder import build_diagnosis_prompt


TOP_CONFLICT_LIMIT = 10


def generate_diagnosis_report(
    metrics_path,
    conflict_log_path,
    output_report_path,
    output_prompt_path,
) -> dict[str, Any]:
    """生成本地占位诊断报告，并保存可发送给 AI 的 prompt。"""
    metrics = _load_metrics(metrics_path)
    conflict_log = _load_conflict_log(conflict_log_path)
    top_conflicts = _select_top_conflicts(conflict_log)
    layout_summary = _placeholder_layout_summary()
    family_summary = _placeholder_family_summary()

    prompt = build_diagnosis_prompt(
        metrics=metrics,
        top_conflicts=top_conflicts,
        layout_summary=layout_summary,
        family_summary=family_summary,
    )

    prompt_path = Path(output_prompt_path)
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    report = _build_placeholder_report(metrics, top_conflicts)
    report_path = Path(output_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    return {
        "metrics": metrics,
        "top_conflicts": top_conflicts,
        "prompt": prompt,
        "prompt_path": str(prompt_path),
        "report_path": str(report_path),
    }


def _load_metrics(path) -> dict:
    metrics_path = Path(path)
    with metrics_path.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics.json must contain an object: {metrics_path}")
    return metrics


def _load_conflict_log(path) -> pd.DataFrame:
    conflict_path = Path(path)
    if not conflict_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(conflict_path)
    except EmptyDataError:
        return pd.DataFrame()


def _select_top_conflicts(conflict_log: pd.DataFrame) -> list[dict]:
    if conflict_log.empty:
        return []

    records = conflict_log.fillna("").to_dict(orient="records")
    type_counts = conflict_log.get("conflict_type", pd.Series(dtype=str)).value_counts()

    def priority(record: dict) -> tuple[int, str]:
        conflict_type = str(record.get("conflict_type", ""))
        return (-int(type_counts.get(conflict_type, 0)), str(record.get("time", "")))

    sorted_records = sorted(records, key=priority)
    return [_json_safe_record(record) for record in sorted_records[:TOP_CONFLICT_LIMIT]]


def _json_safe_record(record: dict) -> dict:
    safe_record = {}
    for key, value in record.items():
        if hasattr(value, "item"):
            value = value.item()
        safe_record[str(key)] = value
    return safe_record


def _placeholder_layout_summary() -> dict:
    return {
        "status": "not_provided",
        "note": "当前本地报告生成函数未接收 layout 输入。后续接入 API 前建议补充房间、家具、活动点、门洞和改造约束摘要。",
    }


def _placeholder_family_summary() -> dict:
    return {
        "status": "not_provided",
        "note": "当前本地报告生成函数未接收 family 输入。后续建议补充家庭成员角色、年龄、行动能力、隐私需求和噪声敏感度摘要。",
    }


def _build_placeholder_report(metrics: dict, top_conflicts: list[dict]) -> str:
    conflict_lines = _conflict_lines(top_conflicts)
    metric_lines = _metric_lines(metrics)

    return f"""# 原户型空间问题诊断报告

> 本报告为本地占位版本，尚未调用大语言模型 API。完整自然语言诊断请使用同目录下保存的 diagnosis prompt 接入模型生成。

## 1. 模拟数据摘要

已读取 metrics.json，并提取到 {len(metrics)} 个顶层指标。

{metric_lines}

## 2. 现有优势与可保留价值

本地占位报告不直接推断完整空间结论，但已将“优势识别”写入 diagnosis prompt。后续 AI 诊断会优先分析当前户型中可保留的功能邻接、较少冲突的生活环节、较顺畅的活动点可达性，以及适合继续强化的照护与休息条件。

如果 conflict_log 中某些冲突类型没有出现，可作为“当前模拟未观察到明显冲突”的初步正向证据；但仍需结合路径日志、活动点设置和户型摘要复核，避免把数据缺失误判为空间优势。

## 3. 主要问题

已从 conflict_log.csv 中选取 {len(top_conflicts)} 条代表性冲突事件，供后续 AI 诊断使用。

{conflict_lines}

## 4. 数据证据

当前占位报告仅整理输入数据，不新增推断结论。后续 AI 诊断应基于 metrics 与 top_conflicts 引用具体指标和冲突记录。

## 5. 空间原因分析

暂未调用 AI，因此不生成具体空间原因。建议在接入 API 时结合户型摘要、活动点、房间功能、家具布置和路径热力图共同解释。

## 6. 优势保留与改造建议

暂未调用 AI，因此不生成最终改造建议。后续建议先识别并保留现有优势，再围绕等待、动线交叉、工作干扰、隐私暴露和老人夜间安全进行设计策略诊断，避免把已经有效的空间关系在改造中过度打散。

## 7. 改造后复测建议

建议复测 metrics.json 中已有指标，并对冲突日志重新进行统计，对比改造前后的路径长度、等待时间、冲突次数和风险事件变化。
"""


def _metric_lines(metrics: dict) -> str:
    if not metrics:
        return "- 暂无指标数据。"
    lines = []
    for key, value in metrics.items():
        lines.append(f"- `{key}`: {value}")
    return "\n".join(lines)


def _conflict_lines(top_conflicts: list[dict]) -> str:
    if not top_conflicts:
        return "- 暂无冲突事件。"
    lines = []
    for index, conflict in enumerate(top_conflicts, start=1):
        conflict_type = conflict.get("conflict_type", "unknown")
        time = conflict.get("time", "")
        agent_id = conflict.get("agent_id", "")
        description = conflict.get("description", "")
        lines.append(
            f"- {index}. `{conflict_type}` 时间: {time} 成员: {agent_id} {description}"
        )
    return "\n".join(lines)


__all__ = ["generate_diagnosis_report"]
