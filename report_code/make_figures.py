"""Draw the report figures from the panel and the saved results.

    Figure 1  output/figures/fig1_motivation.pdf    France and Germany; long differences (step 3)
    Figure 2  output/figures/fig2_estimates.pdf     main estimates by estimator and sample (steps 4-5)
    Figure 3  output/figures/fig3_mechanisms.pdf    poverty lines and household types (step 6)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from style import BLUE, FIG_STYLE, GREY, INK, ORANGE, PANEL, load, save_fig

plt.rcParams.update(FIG_STYLE)


def signed(x, d=2):
    """+0.12 / -0.12, and 0.00 instead of -0.00."""
    s = f"{x:+.{d}f}"
    s = s[1:] if float(s) == 0 else s
    return s.replace("-", "−")                  # typographic minus sign


df = pd.read_csv(PANEL)
s3, s4, s5, s6 = load("step3"), load("step4"), load("step5"), load("step6")["outcomes"]

# --- Figure 1: motivating evidence ------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.98), gridspec_kw={"width_ratios": [1, 1, 1.15]})
fd = pd.DataFrame(s3["fr_de"])
for geo, col, name in [("FR", BLUE, "France"), ("DE", ORANGE, "Germany")]:
    s = fd[fd.geo == geo]
    ax[0].plot(s.year, s.mw_index, "-o", ms=2.5, lw=1.4, color=col, label=name)
    ax[1].plot(s.year, s.iwp_sal, "-o", ms=2.5, lw=1.4, color=col, label=name)
    brk = s[s.brk == 1]
    ax[1].scatter(brk.year, brk.iwp_sal, s=22, facecolor="white", edgecolor=col, zorder=3, lw=1)
for a in ax[:2]:
    a.axvspan(2021.5, 2023.5, color=GREY, alpha=0.15, lw=0)
ax[0].axhline(100, color=GREY, lw=0.6)
ax[0].set(title="A. Real minimum wage (2015 = 100)", xlabel="Year")
ax[1].set(title="B. In-work poverty, employees (%)", xlabel="Income year")
ax[0].legend(frameon=False, loc="upper left")
ld = s3["long_diff"]
pts = pd.DataFrame(ld["points"])
ax[2].scatter(pts.d_mw, pts.d_iwp, s=12, color=BLUE)
for _, r in pts.iterrows():
    ax[2].annotate(r.geo, (r.d_mw, r.d_iwp), xytext=(2, 2), textcoords="offset points", fontsize=5.5, color=INK)
grid = np.linspace(pts.d_mw.min(), pts.d_mw.max(), 20)
ax[2].plot(grid, ld["intercept"] + ld["slope"] * grid, color=ORANGE, lw=1.2)
ax[2].axhline(0, color=GREY, lw=0.6)
ax[2].set(title="C. Changes 2015 to 2024, 21 countries", xlabel="Change in log real minimum wage (x100)",
          ylabel="Change in in-work poverty, pp")
ax[2].text(0.98, 0.95, f"slope {ld['slope']:.3f} (se {ld['se']:.3f})", transform=ax[2].transAxes,
           ha="right", va="top", fontsize=6.5, color=INK)
fig.tight_layout(w_pad=2.2)
save_fig(fig, "fig1_motivation")
plt.close(fig)

# --- Figure 2: estimates by estimator and sample -------------------------------------------
L, nr, chk = s4["ladder"], s4["no_romania"], s5["checks"]
per10 = lambda e: {**e, "b": e["b"] * 0.1, "ci_lo": e["ci_lo"] * 0.1, "ci_hi": e["ci_hi"] * 0.1}
rows = [  # label, real-MW estimate (per +10%), Kaitz estimate, colour
    ("Employees, OLS (main)", per10(L["(4)"]["log_real_mw"]), L["(5)"]["kaitz"], BLUE),
    ("Employees, weighted by employees", chk["wls_employees"]["mw"], chk["wls_employees"]["k"], BLUE),
    ("Employees, FGLS", chk["fgls"]["mw"], chk["fgls"]["k"], BLUE),
    ("Employees, without Romania", per10(nr["(4)"]), nr["(5)"], BLUE),
    ("All employed, OLS", per10(L["(7)"]["log_real_mw"]), L["(8)"]["kaitz"], GREY),
    ("All employed, without Romania", per10(nr["(7)"]), nr["(8)"], GREY),
]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.02), sharey=True)
for ax, (title, j) in zip(axes, [("A. Real minimum wage +10%", 1), ("B. Kaitz index +1 point", 2)]):
    for k, row in enumerate(rows):
        e, col, yk = row[j], row[3], len(rows) - 1 - k
        ax.plot([e["ci_lo"], e["ci_hi"]], [yk, yk], color=col, lw=1.6, solid_capstyle="round")
        ax.scatter([e["b"]], [yk], s=26, color=col, edgecolor="white", lw=1.2, zorder=3)
        ax.annotate(f"{signed(e['b'])} [{e['p_boot']:.2f}]", (e["ci_hi"], yk), xytext=(4, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=6.2, color=INK)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi + 0.3 * (hi - lo))            # room for the labels on the right
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_title(title, loc="left")
    ax.set_xlabel("Change in in-work poverty (pp), 95% CI", fontsize=7)
    ax.grid(axis="x", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_visible(False)
axes[0].set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
axes[0].set_ylim(-0.6, len(rows) - 0.3)
fig.tight_layout()
save_fig(fig, "fig2_estimates")
plt.close(fig)

# --- Figure 3: mechanisms ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.75), gridspec_kw={"width_ratios": [1, 1.25]})
lines = [("40%", "pov_sal_40"), ("50%", "pov_sal_50"), ("60%", "pov_sal_60"), ("70%", "pov_sal_70")]
ax = axes[0]
for k, (lab, key) in enumerate(lines):
    e = s6[key]["log_real_mw"]
    m = e["mean"]
    lo, b, hi = 100 * e["ci_lo"] / m, 100 * e["b"] / m, 100 * e["ci_hi"] / m
    ax.plot([k, k], [lo, hi], color=BLUE, lw=1.6, solid_capstyle="round")
    ax.scatter([k], [b], s=26, color=BLUE, edgecolor="white", lw=1.2, zorder=3)
    ax.annotate(f"{signed(b, 1)}%\n[{e['p_boot']:.2f}]", (k, b), xytext=(5, -3), textcoords="offset points",
                fontsize=6.2, color=INK, va="center")
ax.axhline(0, color=INK, lw=0.6)
ax.set_xticks(range(len(lines)), [f"{lab} of median\n(mean {s6[key]['log_real_mw']['mean']:.1f}%)" for lab, key in lines], fontsize=6.3)
ax.set_xlim(-0.4, len(lines) - 0.3)
ax.set(title="A. Employees, by poverty line (2014-2024)", ylabel="% of the mean rate, per +10%")
groups = [("Full-time", "iwp_ft"), ("Part-time", "iwp_pt"), ("Single adult", "iwp_single"),
          ("Single, children", "iwp_single_kids"), ("2+ adults", "iwp_couple_nokids"),
          ("2+ adults, children", "iwp_couple_kids")]
ax = axes[1]
for k, (lab, key) in enumerate(groups):
    e = s6[key]["log_real_mw"]
    ax.plot([k, k], [e["ci_lo"], e["ci_hi"]], color=GREY if k < 2 else BLUE, lw=1.6, solid_capstyle="round")
    ax.scatter([k], [e["b"]], s=26, color=GREY if k < 2 else BLUE, edgecolor="white", lw=1.2, zorder=3)
    ax.annotate(signed(e["b"]), (k, e["b"]), xytext=(5, 3), textcoords="offset points", fontsize=6.2, color=INK,
                va="center")
ax.axhline(0, color=INK, lw=0.6)
ax.set_xticks(range(len(groups)), [f"{lab}\n({s6[key]['log_real_mw']['mean']:.0f}%)" for lab, key in groups], fontsize=6.3)
ax.set_xlim(-0.4, len(groups) - 0.4)
ax.set(title="B. All employed, by group (pp per +10%)")
for a in axes:
    a.grid(axis="y", color="#e5e5e5", lw=0.6)
    a.set_axisbelow(True)
    a.tick_params(length=0)
fig.tight_layout(w_pad=2.0)
save_fig(fig, "fig3_mechanisms")
plt.close(fig)

print("Figures written to output/figures/")
