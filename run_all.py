"""Reproduce every number, table and figure of the report, and the notebook.

    python run_all.py                 # uses the Eurostat snapshot in data/raw/
    python run_all.py --download      # downloads fresh data from Eurostat first
    python run_all.py --no-notebook   # skips rebuilding analysis.ipynb

Steps: analysis/01-06 (data, panel, estimation; results saved to output/results/),
then report_code/ (LaTeX tables, report and slide figures, notebook). About one minute in total.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANALYSIS = ["01_get_data.py", "02_build_panel.py", "03_descriptives.py", "04_main_regressions.py",
            "05_diagnostics_robustness.py", "06_mechanisms.py"]
REPORT = ["make_tables.py", "make_figures.py", "make_slide_figures.py"] + ([] if "--no-notebook" in sys.argv else ["make_notebook.py"])


def run(folder, script, *args):
    print(f"\n=== {folder}/{script} ===", flush=True)
    subprocess.run([sys.executable, script, *args], cwd=ROOT / folder, check=True)


if __name__ == "__main__":
    for script in ANALYSIS:
        run("analysis", script, *(["--download"] if script == "01_get_data.py" and "--download" in sys.argv else []))
    for script in REPORT:
        run("report_code", script)
    print("\nDone. Numbers in output/results, tables in output/tables, figures in output/figures.")
