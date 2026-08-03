"""Shared publication style and export helpers for the DRLLM figure package.

The palette is semantic rather than "ours versus baseline":
reachability/oracle is indigo; realized/deployed/repair is rose; controls are
neutral; losses and breakage use muted red.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


SINGLE_COLUMN_IN = 3.3125
DOUBLE_COLUMN_IN = 7.0
EXPORT_DPI = 600

PALETTE = {
    "reachability": "#484878",
    "reachability_mid": "#7884B4",
    "reachability_soft": "#B4C0E4",
    "realized": "#C86AA0",
    "realized_mid": "#E0A8C4",
    "realized_soft": "#E4CCD8",
    "control_dark": "#606060",
    "control_mid": "#A8A8A8",
    "control_light": "#D8D8D8",
    "loss": "#B64342",
    "gain": "#2E8B57",
    "teal": "#2F8E84",
    "ink": "#272727",
    "grid": "#E9E9E9",
    "band_lilac": "#EAEAF4",
    "band_aqua": "#E8F2F2",
    "white": "#FFFFFF",
}

SERIF_FALLBACK = [
    "Nimbus Roman No9 L",
    "Nimbus Roman",
    "Times New Roman",
    "Liberation Serif",
    "DejaVu Serif",
]


def _register_serif_fonts() -> None:
    for path in fm.findSystemFonts():
        filename = os.path.basename(path).lower()
        if "nimbusrom" in filename or "nimbus_roman" in filename or filename.startswith("times"):
            try:
                fm.fontManager.addfont(path)
            except RuntimeError:
                pass


def apply_style(base_font_size: float = 8.0) -> None:
    _register_serif_fonts()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": SERIF_FALLBACK,
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": base_font_size,
            "axes.labelsize": base_font_size + 0.5,
            "axes.titlesize": base_font_size + 1.0,
            "axes.titleweight": "bold",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.85,
            "axes.edgecolor": PALETTE["control_dark"],
            "xtick.labelsize": base_font_size - 0.3,
            "ytick.labelsize": base_font_size - 0.3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "legend.fontsize": base_font_size - 0.2,
            "text.color": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
        }
    )


def add_panel_label(
    ax: plt.Axes,
    label: str,
    *,
    x: float = -0.14,
    y: float = 1.04,
    fontsize: float = 11.0,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=PALETTE["ink"],
    )


def quiet_grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(axis=axis, color=PALETTE["grid"], linewidth=0.65, zorder=0)
    ax.set_axisbelow(True)


def _outline_pdf(source: Path, destination: Path) -> bool:
    ghostscript = os.environ.get("GS_BINARY", "gs")
    if shutil.which(ghostscript) is None:
        return False
    args = [
        ghostscript,
        "-q",
        "-dNOPAUSE",
        "-dBATCH",
        "-dSAFER",
        "-sDEVICE=pdfwrite",
        "-dNoOutputFonts",
        "-dAutoRotatePages=/None",
        "-dAutoFilterColorImages=false",
        "-dAutoFilterGrayImages=false",
        "-dColorImageFilter=/FlateEncode",
        "-dGrayImageFilter=/FlateEncode",
        "-dDownsampleColorImages=false",
        "-dDownsampleGrayImages=false",
        "-dDownsampleMonoImages=false",
        "-dColorConversionStrategy=/LeaveColorUnchanged",
        f"-sOutputFile={destination}",
        str(source),
    ]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
        if destination.exists():
            destination.unlink()
        raise RuntimeError(f"Ghostscript PDF outlining failed: {result.stderr[:500]}")
    return True


def save_publication_figure(
    fig: plt.Figure,
    output_base: str | Path,
    *,
    dpi: int = EXPORT_DPI,
    outline_submission_pdf: bool = True,
) -> list[Path]:
    """Export an exact-size editable package plus a submission PDF.

    Do not pass bbox_inches="tight": the figure scripts must manage their own
    margins so the exported page retains the contracted single/double-column
    dimensions.
    """
    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    svg = base.with_suffix(".svg")
    editable_pdf = base.with_name(base.name + "_editable").with_suffix(".pdf")
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    tiff = base.with_suffix(".tiff")

    fig.savefig(svg)
    fig.savefig(editable_pdf)
    fig.savefig(png, dpi=dpi)
    fig.savefig(tiff, dpi=dpi)
    if outline_submission_pdf and _outline_pdf(editable_pdf, pdf):
        pass
    else:
        shutil.copy2(editable_pdf, pdf)
    return [svg, editable_pdf, pdf, png, tiff]

