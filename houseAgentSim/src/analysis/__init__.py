"""指标计算、冲突检测和历史对比分析接口。"""

from src.analysis.comparison import (
    compare_metric_files,
    compare_metrics,
    save_comparison_report_data,
)
from src.analysis.conflicts import (
    detect_bathroom_queue,
    detect_elderly_night_risk,
    detect_privacy_exposure,
    detect_work_interference,
    summarize_conflicts,
)
from src.analysis.metrics import compute_metrics, save_metrics

__all__ = [
    "compare_metric_files",
    "compare_metrics",
    "compute_metrics",
    "detect_bathroom_queue",
    "detect_elderly_night_risk",
    "detect_privacy_exposure",
    "detect_work_interference",
    "save_comparison_report_data",
    "save_metrics",
    "summarize_conflicts",
]
