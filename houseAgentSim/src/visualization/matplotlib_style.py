"""住宅可视化图像共用的 Matplotlib 字体设置。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager


PREFERRED_CHINESE_FONTS = [
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "Arial Unicode MS",
]

WINDOWS_CHINESE_FONT_FILES = [
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/msyhbd.ttc"),
    Path("C:/Windows/Fonts/msyhl.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
]


def _existing_chinese_font_file() -> Path | None:
    for font_path in WINDOWS_CHINESE_FONT_FILES:
        if font_path.exists():
            return font_path
    return None


def _register_windows_chinese_fonts() -> None:
    for font_path in WINDOWS_CHINESE_FONT_FILES:
        if not font_path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
        except RuntimeError:
            continue


def configure_matplotlib_font() -> None:
    """配置中文字体，优先使用 Windows 自带的微软雅黑。"""
    _register_windows_chinese_fonts()
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = PREFERRED_CHINESE_FONTS + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42


def chinese_font_properties() -> font_manager.FontProperties:
    """返回可显式传给文字对象的中文字体属性。"""
    font_file = _existing_chinese_font_file()
    if font_file is not None:
        return font_manager.FontProperties(fname=str(font_file))
    return font_manager.FontProperties(family=PREFERRED_CHINESE_FONTS)


def apply_chinese_font_to_axes(axes) -> None:
    """把中文字体应用到坐标轴、刻度、图例和文本对象。"""
    font_properties = chinese_font_properties()

    axes.title.set_fontproperties(font_properties)
    axes.xaxis.label.set_fontproperties(font_properties)
    axes.yaxis.label.set_fontproperties(font_properties)

    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontproperties(font_properties)

    for text in axes.texts:
        text.set_fontproperties(font_properties)

    legend = axes.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(font_properties)


__all__ = [
    "apply_chinese_font_to_axes",
    "chinese_font_properties",
    "configure_matplotlib_font",
]
