"""Build analysis.ipynb from the analysis scripts and run it.

The scripts in analysis/ are written in cells ("# %% 4(a) Title"), like the
course labs. This script turns each cell into notebook cells: the title and the
comment lines at the top become text, the code becomes a code cell, and the
"# Interpretation:" comment at the end becomes text after the output. The
notebook is then executed from the repository root, so its output is the
output of the same code that produces the report numbers.

    python make_notebook.py            # build and execute (about two minutes)
    python make_notebook.py --no-run   # build only
"""
import ast
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
STEPS = sorted((ROOT / "analysis").glob("0*.py"))

INTRO = """# Minimum wages and in-work poverty in the EU: the analysis

Final project, Econometrics (ECO\\_4MO11), Télécom Paris, November 2026. Yuewen Sun, Victor Zacarias and Robin Herment.

**Question.** Across the 21 EU countries with a statutory minimum wage (2009-2024), is a higher real or relative minimum wage associated with lower in-work poverty among employees, once permanent differences between countries and shocks common to all of them are held fixed?

This notebook is built from the scripts in `analysis/` by `report_code/make_notebook.py` and run from the repository root. Each section is one script. The text before a code cell says what we do and why, and the text after it gives our reading of the output. The report is `report/main.pdf`, and every number in it is written by these scripts to `output/results/`.
"""


def comment_text(lines):
    """'# text' lines -> markdown text (keeps bullets and paragraph breaks)."""
    out = []
    for ln in lines:
        s = ln.strip()
        s = s[1:] if s.startswith("#") else s
        out.append(s[1:] if s.startswith(" ") else s)
    return "\n".join(out).strip()


def split_script(path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    body_start = tree.body[0].end_lineno if doc else 0
    lines = src.splitlines()[body_start:]
    blocks, current = [], {"title": None, "lines": []}
    for ln in lines:
        if ln.startswith("# %%"):
            blocks.append(current)
            current = {"title": ln[4:].strip(), "lines": []}
        else:
            current["lines"].append(ln)
    blocks.append(current)
    return doc, blocks


def cells_for(path):
    doc, blocks = split_script(path)
    first, _, rest = doc.partition("\n")
    title = first.replace("Paths are relative to this file.", "").strip().rstrip(".")
    cells = [new_markdown_cell(f"## {title}\n\n{rest.strip()}\n\n*Source: `analysis/{path.name}`*")]
    for b in blocks:
        lines = b["lines"]
        while lines and not lines[0].strip():
            lines = lines[1:]
        head = []
        if b["title"]:                                   # leading comments = the question
            while lines and lines[0].lstrip().startswith("#"):
                head.append(lines.pop(0))
        interp = []
        for i, ln in enumerate(lines):
            if ln.startswith("# Interpretation:"):
                interp, lines = lines[i + 1:], lines[:i]
                break
        code = "\n".join(lines).strip()
        if b["title"]:
            cells.append(new_markdown_cell(f"### {b['title']}\n\n{comment_text(head)}"))
        if code:
            cells.append(new_code_cell(code))
        if interp:
            cells.append(new_markdown_cell("**Interpretation.** " + comment_text(interp)))
    return cells


def build():
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells.append(new_markdown_cell(INTRO))
    nb.cells.append(new_code_cell("import sys\nsys.path.insert(0, \"analysis\")   # the scripts import tools.py from there"))
    for path in STEPS:
        nb.cells.extend(cells_for(path))
    nb.cells.append(new_markdown_cell(
        "## The tables and figures of the report\n\n"
        "`report_code/make_tables.py` and `report_code/make_figures.py` read the saved results and write "
        "the LaTeX tables and the figures used in `report/main.tex`. No estimation happens there."))
    nb.cells.append(new_code_cell(
        "import subprocess\nfrom IPython.display import Image, display\n\n"
        "for script in [\"make_tables.py\", \"make_figures.py\"]:\n"
        "    subprocess.run([sys.executable, script], cwd=\"report_code\", check=True)\n"
        "for name in [\"fig1_motivation\", \"fig2_estimates\", \"fig3_mechanisms\"]:\n"
        "    display(Image(filename=f\"output/figures/{name}.png\"))"))
    return nb


if __name__ == "__main__":
    if sys.platform == "win32":                        # avoids a harmless zmq warning on Windows
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    nb = build()
    if "--no-run" not in sys.argv:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=900, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(nb, ROOT / "analysis.ipynb")
    print("Wrote analysis.ipynb with", len(nb.cells), "cells")
