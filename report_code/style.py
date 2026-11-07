"""Paths, colours and the figure style shared by the report tables and figures.

The scripts in this folder do no estimation. They read the results that the
analysis scripts (analysis/) saved in output/results/ and turn them into the
LaTeX tables and PDF figures used in report/main.tex.
"""
from pathlib import Path
import json
import time

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "output" / "results"
TABLES = ROOT / "output" / "tables"
FIGURES = ROOT / "output" / "figures"
PANEL = ROOT / "data" / "panel.csv"
for _p in (TABLES, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREY, INK = "#287c8e", "#b04e16", "#8a8a8a", "#333333"

# Serif font close to the Times text of the report, small sizes for a 2-column look.
FIG_STYLE = {"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
             "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
             "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
             "axes.spines.top": False, "axes.spines.right": False}


def load(step):
    return json.loads((RESULTS / f"{step}.json").read_text(encoding="utf-8"))


def retry(fn, *args, **kwargs):
    """OneDrive can lock a file for a moment while it syncs; try again."""
    for attempt in range(5):
        try:
            return fn(*args, **kwargs)
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)


def save_fig(fig, name):
    for ext in ("pdf", "png"):
        retry(fig.savefig, FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")


def write_table(name, tex):
    retry((TABLES / name).write_text, tex, encoding="utf-8")
