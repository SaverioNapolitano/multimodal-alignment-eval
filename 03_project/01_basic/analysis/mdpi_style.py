"""
MDPI/MTI-compliant figure export helpers.

Addresses the journal figure requirements:
- >=600 dpi raster output (MTI minimum).
- RGB (no alpha channel); alpha is flattened onto a white background.
- Sans-serif fonts from the MDPI-recommended set (Arial/Helvetica).
- ASCII hyphen-minus in tick/text labels instead of the Unicode minus sign.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

DPI = 600


def apply() -> None:
    """Apply MDPI-compliant global matplotlib settings. Call once at import time."""
    mpl.rcParams.update(
        {
            "savefig.dpi": DPI,
            "figure.dpi": 150,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.unicode_minus": False,  # hyphen-minus, not U+2212 (MDPI figure rule)
            "font.size": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(path, fig=None, dpi: int = DPI) -> None:
    """Save `fig` (or the current figure) at >=600 dpi as an RGB PNG with no alpha channel."""
    path = Path(path)
    target = fig if fig is not None else plt
    target.savefig(str(path), dpi=dpi, facecolor="white")

    with Image.open(path) as im:
        if im.mode == "RGB":
            rgb = im.copy()
        else:
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            rgb = bg
    rgb.save(path, format="PNG", dpi=(dpi, dpi))
