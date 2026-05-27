# 住宅户型Agent模拟

这是一个面向住宅改造的 Python 项目。项目基于手动标注的住宅户型 JSON，把户型转换为网格级空间模型，并模拟家庭成员一天内的移动、等待、空间占用和冲突事件。

当前版本可以选择户型、居民画像和行为生成方式，运行模拟后查看路径热力图、冲突点图，并调用 AI 生成“家庭成员视角”的居住体验分析。

目前内置一个家庭成员JSON和一个户型JSON作为格式参考。

转载需标注来源。

## 功能概览

- 手动 JSON 户型读取，不依赖 CAD 自动识别。
- 户型几何校验，包括边界、房间、家具、活动点、门洞和约束。
- 网格化空间建模，支持家具阻挡、墙体阻挡和门洞通行。
- Theta* 路径搜索，减少室内路径过度折线化。
- 自写住户智能体模拟，不依赖 Mesa。
- 支持手动行为脚本，也支持硅基流动 API 生成居民日程。
- 输出路径日志、占用日志、冲突日志、指标 JSON 和可视化图像。
- Streamlit 展示界面包含项目设置、户型网格预览、模拟结果和空间问题诊断。
- 空间问题诊断页保留路径热力图、冲突点图和家庭成员视角 AI 分析。

## 项目结构

```text
housing_agent_sim/
  app.py                      # Streamlit 展示界面
  main.py                     # 命令行占位入口
  requirements.txt            # Python 依赖
  data/
    layouts/                  # 手动标注的户型 JSON
    agents/                   # 家庭成员画像 JSON
    schedules/                # 手动行为脚本 JSON
    configs/                  # 可选配置
  outputs/
    logs/                     # 运行后生成 CSV 日志
    metrics/                  # 运行后生成指标 JSON
    figures/                  # 运行后生成 PNG 图像
    reports/                  # 运行后生成 AI prompt / report
  src/
    layout/                   # 户型 schema、读取、几何和校验
    grid/                     # 网格 cell、grid model、grid builder
    pathfinding/              # Theta* 路径搜索和路径工具
    agents/                   # 居民行为、居民对象和 AI 日程生成
    simulation/               # 时间调度、一天模拟和日志保存
    analysis/                 # 冲突检测和指标计算
    visualization/            # 路径热力图和冲突点图
    ai_feedback/              # 家庭成员视角 AI 分析
```

## 安装

建议使用 Python 3.11 或更新版本。

```bash
pip install -r requirements.txt
```

如果要使用 AI 生成行为脚本或家庭成员视角分析，需要在项目根目录创建 `.env` 文件：

```text
SILICONFLOW_API_KEY=your_API_Key
SILICONFLOW_MODEL=your_model_name
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```


## 运行

```bash
python -m streamlit run app.py --server.port 8501 --server.headless true
```

浏览器打开：

```text
http://localhost:8501
```

## 使用流程

1. 把户型 JSON 放入 `data/layouts/`。
2. 把家庭成员 JSON 放入 `data/agents/`。
3. 如使用手动行为脚本，把 schedule JSON 放入 `data/schedules/`。
4. 在`.env`文件中修改相关key及模型名。
5. 启动 Streamlit，进入“项目设置”页。
6. 选择户型、家庭成员和行为驱动方式。
7. 点击“运行模拟”。
8. 在“户型网格预览”查看平面图和网格图。
9. 在“行为模拟结果”查看 `path_log`、`occupancy_log` 和 `conflict_log`。
10. 在“空间问题诊断”查看路径热力图、冲突点图，并可点击生成家庭成员视角 AI 分析。

## 户型 JSON 说明

户型数据来自手动标注，不做 CAD 自动识别。一个户型 JSON 通常包含：

- `boundary`：户型外轮廓。
- `rooms`：房间多边形和容量。
- `walls`：墙体线段，室内墙会被栅格化为不可通行区域。
- `doors`：门洞位置和宽度，用于在墙体上恢复可通行网格。
- `furniture`：家具多边形，可设置为阻挡或增加通行成本。
- `activity_points`：居民行为目标点，行为脚本中的 `target` 必须能映射到这里。
- `constraints`：改造约束、墙厚、承重墙等信息。

项目内提供了可参考的户型文件：

```text
data/layouts/original_layout.json
data/layouts/original_floorplan_001.json
data/layouts/example_layout_with_doors.json
```

## 家庭成员 JSON 说明

家庭成员 JSON 放在 `data/agents/`，核心字段包括：

- `agent_id`：成员唯一 ID。
- `name`：姓名。
- `age`：年龄。
- `role`：角色，例如老人、配偶、护工等。
- `mobility`：行动能力。
- `privacy_need`：隐私需求。
- `noise_sensitivity`：噪声敏感度。
- `current_point`：初始活动点。
- `personality`、`habits`、`needs`、`routine_notes`：AI 生成行为脚本时使用的生活画像。

## 输出文件

运行模拟后会生成：

```text
outputs/logs/path_log.csv
outputs/logs/wait_log.csv
outputs/logs/occupancy_log.csv
outputs/logs/conflict_log.csv
outputs/metrics/current_metrics.json
outputs/figures/path_heatmap.png
outputs/figures/conflict_map.png
outputs/reports/ai_schedule_prompt.md
outputs/reports/ai_generated_schedule.json
outputs/reports/resident_perspective_prompt.md
outputs/reports/resident_perspective_report.md
```


## AI 功能

项目中 AI 只做两件事：

1. 根据户型活动点和居民画像生成一天行为脚本。
2. 在模拟完成后，根据日志和指标生成家庭成员视角的居住体验分析。

AI 不直接生成户型图，也不参与路径搜索、占用判断或冲突检测。所有空间计算仍由本地 Python 模块完成。


