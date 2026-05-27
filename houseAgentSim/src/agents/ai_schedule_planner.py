"""基于居民画像和户型活动点生成 AI 行为脚本。

这里调用硅基流动兼容 OpenAI 的 Chat Completions 接口，让模型输出结构化
JSON 日程。后续模拟仍由本地 Python 逻辑执行，因此 AI 只负责“生成脚本”，
不直接参与寻路、占用和冲突判定。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from src.agents.behavior import Behavior
from src.agents.resident_agent import ResidentAgent
from src.layout.schema import Layout


DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_SILICONFLOW_MODEL = "Pro/zai-org/GLM-4.7"


def build_ai_schedule_prompt(
    layout: Layout,
    agents: list[ResidentAgent],
    day_type: str = "weekday",
) -> str:
    """构造中文提示词，要求大模型只返回可解析的行为 JSON。"""
    payload = {
        "day_type": day_type,
        "layout": {
            "layout_id": layout.layout_id,
            "layout_name": layout.layout_name,
            "unit": layout.unit,
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
                    "capacity": point.capacity,
                }
                for point in layout.activity_points
            ],
        },
        "residents": [
            {
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
                "profile_extras": agent.profile_extras,
            }
            for agent in agents
        ],
    }

    return f"""你是住宅行为模拟的生活脚本生成器。

任务：
根据输入的户型功能点、家庭成员画像、性格、习惯和照护关系，为每个居民生成一天的行为时间表。

必须遵守：
1. 只输出 JSON，不要输出 Markdown、解释文字或代码块。
2. target 必须优先使用输入 activity_points 中的 id，不得编造不存在的功能点。
3. 不得编造不存在的 agent_id。
4. time 使用 HH:MM；duration 使用分钟整数；priority 使用 1 到 5。
5. tags 可使用 morning、night、bathroom、care、care_assist、privacy、elderly_risk、meal、rest、housework、medicine、rehab 等。
6. 半自理老人应体现慢速移动、床边转移、夜间如厕、用餐服药、客厅短时休息等行为。
7. 住家护工应体现从次卧出发，协助老人起身、如厕、洗漱、服药、家务、夜间响应等行为。
8. 护工照护行为应使用 bathroom_assist、master_transfer、care_observation、caregiver_night_response、care_record 等已有功能点。
9. 夫妻二人可以共享 master_bed、sofa、tv、dining_seat 等多人容量点；卫生间如厕、淋浴仍应错峰。
10. 如果输入数据不足，请仍生成保守合理的时间表，并在 notes 中说明缺少哪些信息。

输出 JSON 格式必须为：
{{
  "schedules": {{
    "agent_id": [
      {{
        "time": "07:00",
        "activity": "wash",
        "target": "bathroom_sink",
        "duration": 15,
        "tags": ["morning", "bathroom"],
        "priority": 2
      }}
    ]
  }},
  "notes": ["说明数据不足或生成假设"]
}}

输入数据：
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def parse_ai_schedule_response(response_text: str) -> dict[str, list[Behavior]]:
    """把模型返回文本解析为按 agent_id 分组的 Behavior 列表。"""
    raw_json = _strip_code_fence(response_text.strip())
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AI schedule response is not valid JSON: {exc.msg}") from exc

    if not isinstance(data, Mapping):
        raise ValueError("AI schedule response must be a JSON object.")
    schedules = data.get("schedules")
    if not isinstance(schedules, Mapping):
        raise ValueError("AI schedule response must contain a 'schedules' object.")

    result: dict[str, list[Behavior]] = {}
    for agent_id, behavior_items in schedules.items():
        if not isinstance(agent_id, str):
            raise ValueError("AI schedule keys must be agent ids as strings.")
        if not isinstance(behavior_items, list):
            raise ValueError(f"AI schedule for agent '{agent_id}' must be a list.")
        result[agent_id] = [
            Behavior.from_dict(item, f"schedules.{agent_id}[{index}]")
            for index, item in enumerate(behavior_items)
        ]
    return result


def apply_ai_schedules(
    agents: list[ResidentAgent],
    schedules_by_agent: dict[str, list[Behavior]],
) -> list[str]:
    """把 AI 生成的 schedule 写回居民对象，并返回非致命提示。"""
    warnings: list[str] = []
    agents_by_id = {agent.agent_id: agent for agent in agents}

    for agent in agents:
        generated_schedule = schedules_by_agent.get(agent.agent_id, [])
        agent.schedule = generated_schedule
        if not generated_schedule:
            warnings.append(f"AI did not generate a schedule for agent '{agent.agent_id}'.")

    for agent_id in schedules_by_agent:
        if agent_id not in agents_by_id:
            warnings.append(f"AI generated a schedule for unknown agent '{agent_id}'.")
    return warnings


def generate_ai_schedules(
    layout: Layout,
    agents: list[ResidentAgent],
    model: str | None = None,
    day_type: str = "weekday",
    temperature: float = 0.4,
) -> tuple[dict[str, list[Behavior]], list[str]]:
    """调用硅基流动，根据居民画像生成一天行为脚本。"""
    _load_local_env()
    api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "SILICONFLOW_API_KEY is not set. Set it before using AI-generated "
            "schedules."
        )

    selected_model = (
        model
        or os.getenv("SILICONFLOW_MODEL")
        or os.getenv("OPENAI_MODEL")
        or DEFAULT_SILICONFLOW_MODEL
    )
    base_url = os.getenv("SILICONFLOW_BASE_URL") or DEFAULT_SILICONFLOW_BASE_URL
    prompt = build_ai_schedule_prompt(layout, agents, day_type=day_type)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The 'openai' package is required for AI-generated schedules. "
            "Install dependencies from requirements.txt."
        ) from exc

    # 硅基流动使用 OpenAI 兼容协议，因此这里复用 openai SDK。
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": "你只输出可解析 JSON，不输出 Markdown 或额外解释。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        stream=False,
        extra_body={"enable_thinking": False},
    )
    response_text = _response_text(response)
    schedules = parse_ai_schedule_response(response_text)
    warnings = validate_generated_schedules(layout, agents, schedules)
    return schedules, warnings


def save_ai_schedule_prompt(
    layout: Layout,
    agents: list[ResidentAgent],
    output_path: str | Path,
    day_type: str = "weekday",
) -> str:
    """保存 AI 行为生成 prompt，便于人工检查或手动调用接口。"""
    prompt = build_ai_schedule_prompt(layout, agents, day_type=day_type)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")
    return prompt


def save_ai_schedules(
    schedules_by_agent: dict[str, list[Behavior]],
    output_path: str | Path,
) -> None:
    """把 AI 生成的行为脚本保存为 JSON，便于复核和复现实验。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "schedules": {
            agent_id: [asdict(behavior) for behavior in schedule]
            for agent_id, schedule in schedules_by_agent.items()
        }
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_generated_schedules(
    layout: Layout,
    agents: list[ResidentAgent],
    schedules_by_agent: dict[str, list[Behavior]],
) -> list[str]:
    """校验 AI 日程中的成员和目标点是否能被当前项目识别。"""
    warnings: list[str] = []
    agent_ids = {agent.agent_id for agent in agents}
    accepted_targets = _accepted_target_ids(layout)

    for agent_id, schedule in schedules_by_agent.items():
        if agent_id not in agent_ids:
            warnings.append(f"AI generated schedule for unknown agent '{agent_id}'.")
        for index, behavior in enumerate(schedule):
            if behavior.target not in accepted_targets:
                warnings.append(
                    f"AI schedule {agent_id}[{index}] uses unknown target "
                    f"'{behavior.target}'."
                )
    return warnings


def _accepted_target_ids(layout: Layout) -> set[str]:
    targets = {point.id for point in layout.activity_points}
    targets.update(room.id for room in layout.rooms)
    targets.update(room.type for room in layout.rooms)
    targets.update(
        {
            "master_bedroom",
            "second_bedroom",
            "living_room",
            "dining_room",
            "bathroom",
            "kitchen",
            "balcony",
        }
    )
    return targets


def _strip_code_fence(text: str) -> str:
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL)
    return match.group(1).strip() if match else text


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    try:
        data = response.model_dump()
    except AttributeError:
        data = response if isinstance(response, Mapping) else None

    if isinstance(data, Mapping):
        choices = data.get("choices", [])
        if choices:
            first_choice = choices[0]
            if isinstance(first_choice, Mapping):
                message = first_choice.get("message", {})
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        return content

    try:
        output = response.model_dump().get("output", [])
    except AttributeError:
        output = getattr(response, "output", [])
    chunks: list[str] = []
    for item in output:
        content = item.get("content", []) if isinstance(item, Mapping) else []
        for content_item in content:
            if isinstance(content_item, Mapping) and isinstance(
                content_item.get("text"), str
            ):
                chunks.append(content_item["text"])
    if chunks:
        return "\n".join(chunks)
    raise ValueError("SiliconFlow response did not contain text output.")


def _load_local_env() -> None:
    for env_path in _candidate_env_paths():
        if env_path.exists() and env_path.is_file():
            _load_env_file(env_path)


def _candidate_env_paths() -> list[Path]:
    candidates = [
        Path.cwd() / ".env",
    ]
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            unique_candidates.append(resolved)
            seen.add(resolved)
    return unique_candidates


def _load_env_file(env_path: Path) -> None:
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


__all__ = [
    "apply_ai_schedules",
    "build_ai_schedule_prompt",
    "generate_ai_schedules",
    "parse_ai_schedule_response",
    "save_ai_schedule_prompt",
    "save_ai_schedules",
    "validate_generated_schedules",
]
