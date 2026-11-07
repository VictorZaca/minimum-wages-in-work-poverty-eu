"""Draw the figures used in the presentation slides (PNG, large fonts).

Like make_figures.py, this script does no estimation: it reads the panel and
the results saved by analysis/ and writes output/figures/slides/*.png.

    slide_long_diff.png       changes 2015-2024 in all 21 countries (step 3)
    slide_estimates.png       main estimates by estimator (steps 4-5)
    slide_simulation.png      our estimates against the EU simulation (step 4(e))
    slide_poverty_lines.png   deeper and shallower poverty lines (step 6(a))
    slide_france_germany.png  backup: France and Germany (step 3)
    slide_leave_one_out.png   backup: one country dropped at a time (step 5(h))
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from style import FIGURES, PANEL, load, retry

OUT = FIGURES / "slides"
OUT.mkdir(parents=True, exist_ok=True)

# Colours of the slide deck: paper background, dark ink, teal for our estimates,
# orange for the EU simulation.
PAPER, INK, BODY = "#F6F4EF", "#17212B", "#4A5563"
TEAL, ORANGE, GREY, GRID = "#1F6474", "#D9822B", "#8A8F98", "#E2DED4"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17, "xtick.labelsize": 16, "ytick.labelsize": 16,
    "axes.edgecolor": BODY, "axes.labelcolor": BODY, "xtick.color": BODY, "ytick.color": BODY,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
})


def signed(x, d=2):
    """+0.12 / −0.12 with a typographic minus, and 0.00 instead of −0.00."""
    s = f"{x:+.{d}f}"
    s = s[1:] if float(s) == 0 else s
    return s.replace("-", "−")


def save(fig, name):
    retry(fig.savefig, OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


df = pd.read_csv(PANEL)
s3, s4, s5, s6 = load("step3"), load("step4"), load("step5"), load("step6")["outcomes"]
L, chk = s4["ladder"], s5["checks"]
per10 = lambda e: {**e, "b": e["b"] * 0.1, "ci_lo": e["ci_lo"] * 0.1, "ci_hi": e["ci_hi"] * 0.1}

# --- Changes 2015 to 2024 in all 21 countries -------------------------------------------------
ld = s3["long_diff"]
pts = pd.DataFrame(ld["points"])
fig, ax = plt.subplots(figsize=(10.4, 6.3))
ax.axhline(0, color=GREY, lw=1)
grid = np.linspace(pts.d_mw.min(), pts.d_mw.max(), 20)
ax.plot(grid, ld["intercept"] + ld["slope"] * grid, color=ORANGE, lw=2.5, zorder=2)
ax.scatter(pts.d_mw, pts.d_iwp, s=70, color=TEAL, edgecolor=PAPER, lw=1.2, zorder=3)
nudge = {"LV": (-24, 6), "CZ": (6, 6), "EE": (6, -16), "SK": (6, -16), "ES": (6, 6), "SI": (-24, -16),
         "IE": (6, 6), "PT": (6, -16), "NL": (6, -16), "LU": (6, -16), "BE": (-6, -20), "FR": (-8, 8)}
for _, r in pts.iterrows():
    ax.annotate(r.geo, (r.d_mw, r.d_iwp), xytext=nudge.get(r.geo, (6, 4)), textcoords="offset points",
                fontsize=14, color=BODY)
ax.set(xlabel="Change in the real minimum wage, 2015–2024 (log points × 100)",
       ylabel="Change in in-work poverty (pp)")
ax.grid(color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
save(fig, "slide_long_diff")

# --- Main estimates by estimator ------------------------------------------------------------
nr = s4["no_romania"]
rows = [("OLS (main model)", per10(L["(4)"]["log_real_mw"]), L["(5)"]["kaitz"]),
        ("WLS, weighted by employees", chk["wls_employees"]["mw"], chk["wls_employees"]["k"]),
        ("FGLS, weighted by precision", chk["fgls"]["mw"], chk["fgls"]["k"]),
        ("OLS without Romania", per10(nr["(4)"]), nr["(5)"])]
fig, axes = plt.subplots(1, 2, figsize=(16.6, 4.9), sharey=True)
for ax, (title, j) in zip(axes, [("Real minimum wage +10%", 1), ("Kaitz index +1 point", 2)]):
    for k, row in enumerate(rows):
        e, yk = row[j], len(rows) - 1 - k
        strong = e["p_boot"] < 0.05
        col = TEAL if strong else GREY
        ax.plot([e["ci_lo"], e["ci_hi"]], [yk, yk], color=col, lw=4, solid_capstyle="round")
        ax.scatter([e["b"]], [yk], s=150, color=col, edgecolor=PAPER, lw=2, zorder=3)
        ax.text(1.0, yk, f"{signed(e['b'])}   p = {e['p_boot']:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=16, color=INK, fontweight="bold" if strong else "normal")
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.75 * (hi - lo))            # free column on the right for the labels
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_title(title, loc="left", color=INK, fontweight="bold", pad=12)
    ax.set_xlabel("Change in in-work poverty (pp), 95% interval")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
axes[0].set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
axes[0].set_ylim(-0.6, len(rows) - 0.4)
fig.tight_layout(w_pad=3)
save(fig, "slide_estimates")

# --- Our estimates against the EU simulation --------------------------------------------------
imp = s4["jrc_comparison"]["implied"]
fig, ax = plt.subplots(figsize=(16.6, 4.4))
ax.barh([2], [-1], left=[-12], height=0.42, color=ORANGE)                 # JRC: 12-13% fall
ax.annotate("−12 to −13%", (-13, 2), xytext=(-12, 0), textcoords="offset points", ha="right",
            va="center", fontsize=19, fontweight="bold", color=INK)
for yk, key in [(1, "real minimum wage"), (0, "Kaitz index")]:
    e = imp[key]
    ax.plot([e["ci_lo"], e["ci_hi"]], [yk, yk], color=TEAL, lw=4, solid_capstyle="round")
    ax.scatter([e["b"]], [yk], s=170, color=TEAL, edgecolor=PAPER, lw=2, zorder=3)
    ax.annotate(f"{signed(e['b'], 1)}%", (e["ci_hi"], yk), xytext=(14, 0), textcoords="offset points",
                ha="left", va="center", fontsize=19, fontweight="bold", color=INK)
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks([2, 1, 0], ["EU simulation (JRC 2021)", "Our estimate: real minimum wage", "Our estimate: Kaitz index"])
ax.set_xlim(-17, 9)
ax.set_ylim(-0.6, 2.6)
ax.set_xlabel("Change in in-work poverty (% of its level), our estimates with 95% interval")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
ax.spines["left"].set_visible(False)
ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%".replace("-", "−") if v else "0"))
save(fig, "slide_simulation")

# --- Deeper and shallower poverty lines --------------------------------------------------------
lines = [("40%", "pov_sal_40"), ("50%", "pov_sal_50"), ("60%", "pov_sal_60"), ("70%", "pov_sal_70")]
fig, ax = plt.subplots(figsize=(8.4, 6.0))
for k, (lab, key) in enumerate(lines):
    e = s6[key]["log_real_mw"]
    m = e["mean"]
    lo, b, hi = 100 * e["ci_lo"] / m, 100 * e["b"] / m, 100 * e["ci_hi"] / m
    ax.plot([k, k], [lo, hi], color=TEAL, lw=4, solid_capstyle="round")
    ax.scatter([k], [b], s=170, color=TEAL, edgecolor=PAPER, lw=2, zorder=3)
    ax.annotate(f"{signed(b, 0)}%", (k, b), xytext=(14, 0), textcoords="offset points", fontsize=18,
                fontweight="bold", color=INK, va="center")
ax.axhline(0, color=INK, lw=1.2)
ax.set_xticks(range(len(lines)), [f"{lab}\n(rate {s6[key]['log_real_mw']['mean']:.1f}%)" for lab, key in lines])
ax.set_xlim(-0.4, len(lines) - 0.4)
ax.set_xlabel("Poverty line, % of the national median")
ax.set_ylabel("Change in the poverty rate (% of its level)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%".replace("-", "−") if v else "0"))
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(length=0)
save(fig, "slide_poverty_lines")

# --- Backup: France and Germany -----------------------------------------------------------------
fd = pd.DataFrame(s3["fr_de"])
fig, ax = plt.subplots(1, 2, figsize=(16.6, 5.4))
for geo, col, name in [("FR", TEAL, "France"), ("DE", ORANGE, "Germany")]:
    s = fd[fd.geo == geo]
    ax[0].plot(s.year, s.mw_index, "-o", ms=7, lw=3, color=col, label=name)
    ax[1].plot(s.year, s.iwp_sal, "-o", ms=7, lw=3, color=col, label=name)
    brk = s[s.brk == 1]
    ax[1].scatter(brk.year, brk.iwp_sal, s=120, facecolor=PAPER, edgecolor=col, zorder=3, lw=2.5)
for a in ax:
    a.axvspan(2021.5, 2023.5, color=GREY, alpha=0.15, lw=0)
    a.grid(color=GRID, lw=0.8)
    a.set_axisbelow(True)
    a.tick_params(length=0)
ax[0].axhline(100, color=GREY, lw=1)
ax[0].set_title("Real minimum wage (2015 = 100)", loc="left", fontweight="bold")
ax[1].set_title("In-work poverty, employees (%). Hollow = series break", loc="left", fontweight="bold")
ax[0].legend(frameon=False, loc="upper left")
fig.tight_layout(w_pad=3)
save(fig, "slide_france_germany")

# --- Backup: leave one country out ------------------------------------------------------------
loo = pd.DataFrame(s5["leave_one_out"])
fig, ax = plt.subplots(1, 2, figsize=(16.6, 5.0), sharex=True)
for a, v, base, title in [(ax[0], "mw", L["(4)"]["log_real_mw"]["b"] * 0.1, "Real minimum wage +10%"),
                          (ax[1], "k", L["(5)"]["kaitz"]["b"], "Kaitz index +1 point")]:
    a.errorbar(loo.dropped, loo[v], yerr=stats.t.ppf(0.975, 19) * loo[v + "_se"], fmt="o", ms=7, color=TEAL, capsize=0, lw=2.5)
    a.axhline(base, color=ORANGE, lw=2.5, label="all 21 countries")
    a.axhline(0, color=INK, lw=1)
    a.set_title(f"{title}: one country dropped", loc="left", fontweight="bold")
    a.tick_params(axis="x", labelsize=13, rotation=90, length=0)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
ax[0].set_ylabel("pp, 95% interval")
ax[0].legend(frameon=False, loc="upper left")
fig.tight_layout(w_pad=3)
save(fig, "slide_leave_one_out")
print("Slide figures written to output/figures/slides/")
