"""Shared plotting helpers for report scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


PREFERRED_CHINESE_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
    "WenQuanYi Micro Hei",
]


COMMON_CHINESE_FONT_FILES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def configure_chinese_matplotlib() -> Optional[str]:
    """Configure matplotlib so generated charts render Chinese text clearly."""
    import matplotlib as mpl
    from matplotlib import font_manager

    available_names = {font.name for font in font_manager.fontManager.ttflist}
    registered_names: list[str] = []

    for raw_path in COMMON_CHINESE_FONT_FILES:
        font_path = Path(raw_path)
        if not font_path.exists():
            continue
        try:
            font_manager.fontManager.addfont(str(font_path))
            registered_names.append(font_manager.FontProperties(fname=str(font_path)).get_name())
        except Exception:
            continue

    candidates = registered_names + [name for name in PREFERRED_CHINESE_FONTS if name in available_names]
    fallback_fonts = ["DejaVu Sans"]
    font_stack = list(dict.fromkeys(candidates + PREFERRED_CHINESE_FONTS + fallback_fonts))

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = font_stack
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    return candidates[0] if candidates else None
