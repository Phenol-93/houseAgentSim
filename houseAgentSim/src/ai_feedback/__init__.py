"""AI 反馈、诊断 prompt 和成员视角报告接口。"""

from src.ai_feedback.prompt_builder import build_diagnosis_prompt
from src.ai_feedback.report_generator import generate_diagnosis_report
from src.ai_feedback.resident_perspective import (
    build_resident_perspective_prompt,
    generate_resident_perspective_report,
)

__all__ = [
    "build_diagnosis_prompt",
    "build_resident_perspective_prompt",
    "generate_diagnosis_report",
    "generate_resident_perspective_report",
]
