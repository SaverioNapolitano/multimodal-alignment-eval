"""
Frontiers-compliant figure export helpers.

Addresses the journal figure requirements:
- Vector output (EPS/PDF/SVG) for line art such as charts and heatmaps, which
  Frontiers prefers; falls back to >=300 dpi raster (TIFF/JPEG/PNG) when a raster
  format is requested. 600 dpi is used by default (>= the 300 dpi minimum).
- RGB (no alpha channel); alpha is flattened onto a white background for raster.
- Sans-serif fonts (Arial/Helvetica).
- ASCII hyphen-minus in tick/text labels instead of the Unicode minus sign.
"""

import io
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

DPI = 300  # Frontiers submission portal requirement (>= 300 dpi at final size)


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
    """Save `fig` (or the current figure) at >=600 dpi as an RGB image with no alpha.

    Output format is chosen from the file extension. Frontiers accepts JPEG, TIFF and
    EPS for the separately uploaded high-resolution figure files (not PNG); PNG remains
    fine for the figure embedded in the review PDF. The figure is rendered once to a
    lossless in-memory PNG, flattened onto white, then encoded a single time to the
    target format. JPEG uses quality=95 with no chroma subsampling (4:4:4) to limit
    artifacts on text and line art.
    """
    path = Path(path)
    target = fig if fig is not None else plt

    ext = path.suffix.lower()
    if ext in (".eps", ".pdf", ".svg"):
        # Vector output: write directly, preserving the exact figure size (do NOT use
        # bbox_inches="tight", which would alter the 85 mm / 180 mm column widths).
        target.savefig(str(path), format=ext.lstrip("."), facecolor="white")
        return

    buf = io.BytesIO()
    target.savefig(buf, format="png", dpi=dpi, facecolor="white")
    buf.seek(0)
    with Image.open(buf) as im:
        if im.mode == "RGB":
            rgb = im.copy()
        else:
            rgba = im.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[-1])
            rgb = bg

    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        rgb.save(path, format="JPEG", dpi=(dpi, dpi), quality=95, subsampling=0, optimize=True)
    elif ext in (".tif", ".tiff"):
        rgb.save(path, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
    else:
        rgb.save(path, format="PNG", dpi=(dpi, dpi))
