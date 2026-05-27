"""住宅行为模拟系统的 Streamlit 展示入口。

这个文件主要负责把“数据选择、模拟运行、结果展示、AI 分析”串成一个
课程设计可演示的网页界面。核心算法仍放在 ``src/`` 下面，避免界面逻辑
和模拟逻辑混在一起。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.agents.agent_loader import load_agent_profiles, load_agents
from src.agents.ai_schedule_planner import (
    apply_ai_schedules,
    generate_ai_schedules,
    save_ai_schedule_prompt,
    save_ai_schedules,
)
from src.ai_feedback.resident_perspective import generate_resident_perspective_report
from src.analysis.conflicts import (
    detect_bathroom_queue,
    detect_elderly_night_risk,
)
from src.analysis.metrics import compute_metrics, save_metrics
from src.grid.grid_builder import build_grid
from src.grid.grid_model import Grid
from src.layout.loader import load_layout
from src.layout.schema import Layout
from src.layout.validator import validate_layout
from src.simulation.logger import save_all_logs
from src.simulation.simulation_model import HousingSimulation
from src.visualization.conflict_map import plot_conflict_map
from src.visualization.matplotlib_style import (
    apply_chinese_font_to_axes,
    chinese_font_properties,
    configure_matplotlib_font,
)
from src.visualization.path_heatmap import plot_path_heatmap


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
LAYOUT_DIR = DATA_DIR / "layouts"
AGENT_DIR = DATA_DIR / "agents"
SCHEDULE_DIR = DATA_DIR / "schedules"
LOG_DIR = OUTPUT_DIR / "logs"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

configure_matplotlib_font()


def main() -> None:
    """启动 Streamlit 页面并组织四个主要展示页签。"""
    st.set_page_config(page_title="housing_agent_sim", layout="wide")
    st.title("住宅户型行为模拟系统")
    st.caption("基于手动标注 JSON、网格模型和住户智能体的课程设计展示界面")

    _ensure_output_dirs()

    state = st.session_state
    state.setdefault("layout", None)
    state.setdefault("grid", None)
    state.setdefault("simulation_logs", None)
    state.setdefault("run_warnings", [])
    state.setdefault("selected_layout_path", str(LAYOUT_DIR / "original_layout.json"))
    state.setdefault("selected_agent_path", str(AGENT_DIR / "retired_couple_jinan.json"))
    state.setdefault("agents", None)

    tab_settings, tab_grid, tab_results, tab_diagnosis = st.tabs(
        [
            "1. 项目设置",
            "2. 户型网格预览",
            "3. 行为模拟结果",
            "4. 空间问题诊断",
        ]
    )

    with tab_settings:
        _render_project_settings()

    with tab_grid:
        _render_grid_preview(state.get("layout"), state.get("grid"))

    with tab_results:
        _render_simulation_results()

    with tab_diagnosis:
        _render_diagnosis(state.get("layout"), state.get("grid"))


def _render_project_settings() -> None:
    """项目设置页：选择输入文件、行为来源和模拟时间步。"""
    st.header("项目设置")

    layout_files = _json_options(
        LAYOUT_DIR,
        preferred=["original_layout.json", "renovated_layout.json"],
    )
    agent_files = _json_options(AGENT_DIR)
    schedule_files = _json_options(SCHEDULE_DIR)

    behavior_mode = st.radio(
        "行为驱动方式",
        ["AI 生成行为脚本", "手动行为脚本"],
        horizontal=True,
    )
    use_ai_schedule = behavior_mode == "AI 生成行为脚本"

    col_layout, col_agent, col_schedule, col_step = st.columns([1.4, 1.2, 1.2, 0.7])
    with col_layout:
        layout_name = st.selectbox("户型 JSON", layout_files, index=0)
    with col_agent:
        agent_name = st.selectbox("家庭成员 JSON", agent_files, index=0)
    with col_schedule:
        if use_ai_schedule:
            ai_model = st.text_input("硅基流动模型", value="")
            schedule_name = None
        else:
            schedule_name = st.selectbox("行为脚本 JSON", schedule_files, index=0)
            ai_model = ""
    with col_step:
        time_step = st.number_input("time_step", min_value=1, max_value=60, value=5)

    layout_path = LAYOUT_DIR / layout_name
    agent_path = AGENT_DIR / agent_name
    schedule_path = SCHEDULE_DIR / schedule_name if schedule_name else None
    st.session_state.selected_layout_path = str(layout_path)
    st.session_state.selected_agent_path = str(agent_path)

    st.write("当前选择：")
    st.code(
        "\n".join(
            [
                f"layout:   {layout_path}",
                f"agents:   {agent_path}",
                f"behavior: {behavior_mode}",
                f"schedule: {schedule_path or '(AI generated from resident profiles)'}",
                f"model:    {ai_model or 'SILICONFLOW_MODEL / Pro/zai-org/GLM-4.7'}",
                f"time_step: {time_step} minutes",
            ]
        )
    )

    if st.button("运行模拟", type="primary"):
        _run_simulation(
            layout_path,
            agent_path,
            schedule_path,
            int(time_step),
            use_ai_schedule=use_ai_schedule,
            ai_model=ai_model.strip() or None,
        )

    _render_run_messages()


def _run_simulation(
    layout_path: Path,
    agent_path: Path,
    schedule_path: Path | None,
    time_step: int,
    use_ai_schedule: bool = False,
    ai_model: str | None = None,
) -> None:
    """执行完整模拟流程，并把日志、指标和图像写入 outputs。"""
    st.session_state.run_warnings = []
    required_paths = [layout_path, agent_path]
    if not use_ai_schedule:
        if schedule_path is None:
            st.error("手动行为脚本模式需要选择 schedule JSON。")
            return
        required_paths.append(schedule_path)

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]
    if missing:
        st.error("以下输入文件不存在，暂不能运行模拟：")
        st.code("\n".join(missing))
        return

    # 进度条按照真实处理阶段拆分，方便用户看到 AI 调用或模拟运行卡在哪一步。
    steps = [
        "读取户型 JSON",
        "校验户型几何",
        "生成户型网格",
    ]
    if use_ai_schedule:
        steps.extend(
            [
                "读取居民画像",
                "保存 AI 行为生成 prompt",
                "调用硅基流动生成行为脚本",
                "保存 AI 生成的行为脚本",
                "应用 AI 行为脚本",
            ]
        )
    else:
        steps.append("读取手动行为脚本")
    steps.extend(
        [
            "运行一天行为模拟",
            "保存模拟日志",
            "计算并保存指标",
            "生成可视化图像",
            "更新页面状态",
        ]
    )

    progress_bar = st.progress(0)
    status_box = st.empty()
    completed_steps = 0
    total_steps = len(steps)

    def start_step(message: str) -> None:
        status_box.info(f"{completed_steps + 1}/{total_steps} 正在{message}...")

    def finish_step(message: str) -> None:
        nonlocal completed_steps
        completed_steps += 1
        progress_bar.progress(completed_steps / total_steps)
        status_box.success(f"{completed_steps}/{total_steps} 已完成：{message}")

    try:
        start_step("读取户型 JSON")
        layout = load_layout(layout_path)
        finish_step("读取户型 JSON")

        start_step("校验户型几何")
        validation_issues = validate_layout(layout)
        finish_step("校验户型几何")

        start_step("生成户型网格")
        grid, grid_warnings = build_grid(layout)
        finish_step("生成户型网格")

        if use_ai_schedule:
            start_step("读取居民画像")
            agents, agent_warnings = load_agent_profiles(agent_path)
            finish_step("读取居民画像")

            start_step("保存 AI 行为生成 prompt")
            save_ai_schedule_prompt(
                layout,
                agents,
                REPORT_DIR / "ai_schedule_prompt.md",
            )
            finish_step("保存 AI 行为生成 prompt")

            start_step("调用硅基流动生成行为脚本")
            schedules_by_agent, ai_warnings = generate_ai_schedules(
                layout,
                agents,
                model=ai_model,
            )
            finish_step("调用硅基流动生成行为脚本")

            start_step("保存 AI 生成的行为脚本")
            save_ai_schedules(
                schedules_by_agent,
                REPORT_DIR / "ai_generated_schedule.json",
            )
            finish_step("保存 AI 生成的行为脚本")

            start_step("应用 AI 行为脚本")
            agent_warnings.extend(ai_warnings)
            agent_warnings.extend(apply_ai_schedules(agents, schedules_by_agent))
            finish_step("应用 AI 行为脚本")
        else:
            start_step("读取手动行为脚本")
            agents, agent_warnings = load_agents(agent_path, schedule_path)
            finish_step("读取手动行为脚本")

        start_step("运行一天行为模拟")
        simulation = HousingSimulation(layout, grid, agents, time_step=time_step)
        logs = simulation.run_day()
        _append_detected_conflicts(logs, layout, agents)
        finish_step("运行一天行为模拟")

        start_step("保存模拟日志")
        save_all_logs(logs, LOG_DIR)
        finish_step("保存模拟日志")

        start_step("计算并保存指标")
        metrics = compute_metrics(LOG_DIR)
        save_metrics(metrics, METRICS_DIR / "current_metrics.json")
        finish_step("计算并保存指标")

        start_step("生成可视化图像")
        conflict_log = _read_csv(LOG_DIR / "conflict_log.csv")
        if not conflict_log.empty:
            plot_conflict_map(layout, conflict_log, FIGURE_DIR / "conflict_map.png")
        path_log = _read_csv(LOG_DIR / "path_log.csv")
        plot_path_heatmap(grid, path_log, layout, FIGURE_DIR / "path_heatmap.png")
        finish_step("生成可视化图像")

        start_step("更新页面状态")
        st.session_state.layout = layout
        st.session_state.grid = grid
        st.session_state.selected_layout_path = str(layout_path)
        st.session_state.selected_agent_path = str(agent_path)
        st.session_state.agents = agents
        st.session_state.simulation_logs = logs
        st.session_state.run_warnings = (
            validation_issues + grid_warnings + agent_warnings
        )
        finish_step("更新页面状态")
        st.success("模拟已完成，日志、指标和图像已写入 outputs。")
    except Exception as exc:  # pragma: no cover - Streamlit boundary
        st.error(f"运行模拟失败：{exc}")


def _render_run_messages() -> None:
    warnings = st.session_state.get("run_warnings", [])
    if warnings:
        with st.expander("运行提示 / 数据校验问题", expanded=True):
            for warning in warnings:
                st.warning(warning)

    if not any((LAYOUT_DIR / name).exists() for name in _json_options(LAYOUT_DIR)):
        st.info("data/layouts 中尚未发现可运行的户型 JSON。可以先使用已有 outputs 文件查看展示页。")


def _append_detected_conflicts(
    logs: dict[str, list[dict[str, Any]]],
    layout: Layout,
    agents,
) -> None:
    """把分析模块识别到的冲突补充进模拟冲突日志。"""
    conflict_log = logs.setdefault("conflict_log", [])
    existing_keys = {
        (
            record.get("time"),
            record.get("agent_id"),
            record.get("conflict_type"),
            record.get("target"),
        )
        for record in conflict_log
    }

    detected_events: list[dict[str, Any]] = []
    detected_events.extend(
        detect_bathroom_queue(logs.get("wait_log", []), layout.activity_points)
    )
    for agent in agents:
        if agent.role.lower() in {"elder", "elderly", "senior"} or agent.age >= 60:
            detected_events.extend(
                detect_elderly_night_risk(
                    logs.get("path_log", []),
                    elderly_agent_id=agent.agent_id,
                )
            )

    for event in detected_events:
        key = (
            event.get("time"),
            event.get("agent_id"),
            event.get("conflict_type"),
            event.get("target"),
        )
        if key in existing_keys:
            continue
        conflict_log.append(event)
        existing_keys.add(key)


def _render_grid_preview(layout: Layout | None, grid: Grid | None) -> None:
    """户型预览页：展示当前 JSON 对应的平面图和网格图。"""
    st.header("户型网格预览")

    layout, grid = _load_preview_layout_and_grid()
    _render_layout_source_caption()
    warnings = st.session_state.get("preview_grid_warnings", [])
    if warnings:
        st.info("网格生成提示：" + "；".join(warnings))

    if layout is None:
        st.info("尚未载入户型。请在“项目设置”页选择 JSON 并运行模拟。")
        return

    col_plan, col_grid = st.columns(2)
    with col_plan:
        st.subheader("户型平面图")
        st.pyplot(_draw_layout_figure(layout))
    with col_grid:
        st.subheader("网格图")
        if grid is None:
            st.info("尚未生成网格。")
        else:
            st.pyplot(_draw_grid_figure(grid, layout))


def _load_preview_layout_and_grid() -> tuple[Layout | None, Grid | None]:
    selected_layout = _selected_layout_path()
    if selected_layout is None:
        return None, None

    try:
        layout = load_layout(selected_layout)
        grid, warnings = build_grid(layout)
        st.session_state.preview_grid_warnings = warnings
        st.session_state.preview_layout_path = str(selected_layout)
        return layout, grid
    except Exception as exc:
        st.session_state.preview_grid_warnings = [
            f"无法从 data/layouts 自动载入预览：{exc}"
        ]
        return None, None


def _selected_layout_path() -> Path | None:
    selected = st.session_state.get("selected_layout_path")
    if selected:
        selected_path = Path(selected)
        if selected_path.exists():
            return selected_path

    original_path = LAYOUT_DIR / "original_layout.json"
    if original_path.exists():
        return original_path
    return _first_existing_json(LAYOUT_DIR)


def _selected_agent_path() -> Path | None:
    selected = st.session_state.get("selected_agent_path")
    if selected:
        selected_path = Path(selected)
        if selected_path.exists():
            return selected_path

    preferred_path = AGENT_DIR / "retired_couple_jinan.json"
    if preferred_path.exists():
        return preferred_path
    return _first_existing_json(AGENT_DIR)


def _load_current_agents_for_perspective() -> list:
    agents = st.session_state.get("agents")
    if agents:
        return list(agents)

    agent_path = _selected_agent_path()
    if agent_path is None:
        return []
    loaded_agents, _warnings = load_agent_profiles(agent_path)
    return loaded_agents


def _render_simulation_results() -> None:
    """结果页：直接展示核心 CSV 日志，便于调试和课程截图。"""
    st.header("行为模拟结果")
    path_log = _read_csv(LOG_DIR / "path_log.csv")
    occupancy_log = _read_csv(LOG_DIR / "occupancy_log.csv")
    conflict_log = _read_csv(LOG_DIR / "conflict_log.csv")

    result_tabs = st.tabs(["path_log", "occupancy_log", "conflict_log"])
    with result_tabs[0]:
        _show_dataframe_or_empty(path_log, "暂无 path_log。")
    with result_tabs[1]:
        _show_dataframe_or_empty(occupancy_log, "暂无 occupancy_log。")
    with result_tabs[2]:
        _show_dataframe_or_empty(conflict_log, "暂无 conflict_log。")


def _render_diagnosis(layout: Layout | None, grid: Grid | None) -> None:
    """空间问题诊断页：只保留热力图、冲突图和成员视角 AI 分析。"""
    st.header("空间问题诊断")
    layout, grid = _load_preview_layout_and_grid()
    _render_layout_source_caption()

    col_heatmap, col_conflict = st.columns(2)
    with col_heatmap:
        st.subheader("路径热力图")
        heatmap_path = FIGURE_DIR / "path_heatmap.png"
        if layout is not None and grid is not None:
            path_log = _read_csv(LOG_DIR / "path_log.csv")
            if path_log.empty:
                st.info("暂无 path_log，无法生成路径热力图。")
            else:
                plot_path_heatmap(grid, path_log, layout, heatmap_path)
                _show_png(heatmap_path)
        elif heatmap_path.exists():
            _show_png(heatmap_path)
        else:
            st.info("暂无路径热力图。")

    with col_conflict:
        st.subheader("冲突点图")
        conflict_map_path = FIGURE_DIR / "conflict_map.png"
        if layout is not None:
            conflict_log = _read_csv(LOG_DIR / "conflict_log.csv")
            if conflict_log.empty:
                st.info("暂无 conflict_log，无法生成冲突点图。")
            else:
                plot_conflict_map(layout, conflict_log, conflict_map_path)
                _show_png(conflict_map_path)
        elif conflict_map_path.exists():
            _show_png(conflict_map_path)
        else:
            st.info("暂无冲突点图。")

    _render_resident_perspective_analysis(layout)


def _render_resident_perspective_analysis(layout: Layout | None) -> None:
    """调用大模型，从每位家庭成员视角生成居住体验分析。"""
    st.subheader("家庭成员视角 AI 分析")
    st.caption("模拟完成后可单独点击生成，让 AI 站在每位家庭成员的视角评价当前户型体验。")

    report_path = REPORT_DIR / "resident_perspective_report.md"
    prompt_path = REPORT_DIR / "resident_perspective_prompt.md"
    metrics_path = METRICS_DIR / "current_metrics.json"

    col_button, col_model = st.columns([0.8, 1.2])
    with col_model:
        model = st.text_input(
            "成员视角分析模型",
            value="",
            placeholder="留空则读取 .env 中的 SILICONFLOW_MODEL",
        )
    with col_button:
        run_analysis = st.button("生成家庭成员视角分析")

    if run_analysis:
        if layout is None:
            st.error("尚未载入户型，无法生成成员视角分析。")
        elif not metrics_path.exists():
            st.error("尚未发现 outputs/metrics/current_metrics.json，请先运行一次模拟。")
        elif not (LOG_DIR / "path_log.csv").exists():
            st.error("尚未发现模拟日志，请先运行一次模拟。")
        else:
            try:
                agents = _load_current_agents_for_perspective()
                if not agents:
                    st.error("尚未找到家庭成员 JSON，无法生成成员视角分析。")
                    return
                with st.spinner("正在调用硅基流动生成家庭成员视角分析..."):
                    generate_resident_perspective_report(
                        layout=layout,
                        agents=agents,
                        metrics_path=metrics_path,
                        log_dir=LOG_DIR,
                        output_report_path=report_path,
                        output_prompt_path=prompt_path,
                        model=model.strip() or None,
                    )
                st.success("家庭成员视角分析已生成。")
            except Exception as exc:  # pragma: no cover - Streamlit boundary
                st.error(f"生成家庭成员视角分析失败：{exc}")

    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("还没有成员视角分析报告。运行模拟后点击上方按钮即可生成。")

    if prompt_path.exists():
        with st.expander("查看 resident perspective prompt"):
            st.code(prompt_path.read_text(encoding="utf-8"), language="markdown")


def _draw_layout_figure(layout: Layout):
    """绘制户型平面预览图，使用中文字体显示房间名。"""
    configure_matplotlib_font()
    font_properties = chinese_font_properties()
    figure, axes = plt.subplots(figsize=(7, 5))
    _draw_polygon(axes, layout.boundary, edge="#111111", face="#ffffff", linewidth=1.4)
    for room in layout.rooms:
        _draw_polygon(axes, room.polygon, edge="#333333", face="#f7f7f7", linewidth=1.0)
        center = _polygon_center(room.polygon)
        axes.text(
            center[0],
            center[1],
            room.name,
            ha="center",
            va="center",
            fontsize=8,
            fontproperties=font_properties,
        )
    for furniture in layout.furniture:
        _draw_polygon(
            axes,
            furniture.polygon,
            edge="#607d8b",
            face="#d9e5ea",
            linewidth=0.8,
            linestyle="--",
        )
    if layout.activity_points:
        axes.scatter(
            [point.position[0] for point in layout.activity_points],
            [point.position[1] for point in layout.activity_points],
            s=40,
            c="#d62728",
            marker=".",
            label="activity point",
        )
        axes.legend(fontsize=8)
    axes.set_aspect("equal")
    axes.invert_yaxis()
    axes.set_title(layout.layout_name, fontproperties=font_properties)
    apply_chinese_font_to_axes(axes)
    figure.tight_layout()
    return figure


def _draw_grid_figure(grid: Grid, layout: Layout | None = None):
    """绘制网格模型预览，灰色表示不可通行网格。"""
    configure_matplotlib_font()
    font_properties = chinese_font_properties()
    figure, axes = plt.subplots(figsize=(7, 5))
    for row in range(grid.height):
        for col in range(grid.width):
            cell = grid.get_cell(row, col)
            if cell is None:
                continue
            x, y = grid.grid_to_world(row, col)
            lower_left = (
                x - grid.grid_size / 2,
                y - grid.grid_size / 2,
            )
            face = "#d0d0d0" if cell.blocked or not cell.walkable else "#ffffff"
            rect = plt.Rectangle(
                lower_left,
                grid.grid_size,
                grid.grid_size,
                facecolor=face,
                edgecolor="#e0e0e0",
                linewidth=0.5,
            )
            axes.add_patch(rect)
    if layout is not None:
        for room in layout.rooms:
            _draw_outline(axes, room.polygon, color="#333333")
        for furniture in layout.furniture:
            _draw_outline(axes, furniture.polygon, color="#607d8b", linestyle="--")
        for point in layout.activity_points:
            axes.scatter([point.position[0]], [point.position[1]], c="#d62728", s=18)
    axes.set_aspect("equal")
    axes.invert_yaxis()
    axes.set_title("网格预览", fontproperties=font_properties)
    apply_chinese_font_to_axes(axes)
    axes.autoscale()
    figure.tight_layout()
    return figure


def _draw_polygon(
    axes,
    points: list[tuple[float, float]],
    edge: str,
    face: str,
    linewidth: float,
    linestyle: str = "-",
) -> None:
    if not points:
        return
    polygon = plt.Polygon(
        points,
        closed=True,
        edgecolor=edge,
        facecolor=face,
        linewidth=linewidth,
        linestyle=linestyle,
    )
    axes.add_patch(polygon)


def _draw_outline(
    axes,
    points: list[tuple[float, float]],
    color: str,
    linestyle: str = "-",
) -> None:
    if not points:
        return
    closed = points + [points[0]]
    axes.plot(
        [point[0] for point in closed],
        [point[1] for point in closed],
        color=color,
        linewidth=0.9,
        linestyle=linestyle,
    )


def _polygon_center(points: list[tuple[float, float]]) -> tuple[float, float]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _json_options(directory: Path, preferred: list[str] | None = None) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    preferred = preferred or []
    existing = sorted(path.name for path in directory.glob("*.json"))
    options = []
    for name in preferred:
        if name not in options:
            options.append(name)
    for name in existing:
        if name not in options:
            options.append(name)
    return options or ["未找到 JSON 文件"]


def _first_existing_json(directory: Path) -> Path | None:
    for path in sorted(directory.glob("*.json")):
        return path
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _show_dataframe_or_empty(data_frame: pd.DataFrame, empty_message: str) -> None:
    if data_frame.empty:
        st.info(empty_message)
    else:
        st.dataframe(data_frame, use_container_width=True)


def _render_layout_source_caption() -> None:
    layout_path_value = st.session_state.get("preview_layout_path")
    if not layout_path_value:
        return
    layout_path = Path(layout_path_value)
    if not layout_path.exists():
        st.caption(f"当前户型来源：{layout_path}")
        return
    updated_at = datetime.fromtimestamp(layout_path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    st.caption(f"当前户型来源：{layout_path}；JSON 修改时间：{updated_at}")


def _show_png(path: Path) -> None:
    if not path.exists():
        st.info(f"图像文件不存在：{path}")
        return
    updated_at = datetime.fromtimestamp(path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    st.caption(f"图像更新时间：{updated_at}")
    st.image(path.read_bytes(), use_container_width=True)


def _ensure_output_dirs() -> None:
    for directory in (LOG_DIR, METRICS_DIR, FIGURE_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
