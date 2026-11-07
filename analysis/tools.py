"""Shared set-up for the analysis scripts in this folder.

Paths are relative to this file, so the scripts run from any working directory.
The sample, the regression specification, the cluster-robust standard errors
and the wild cluster bootstrap are defined once here and used by every step.

Numbers quoted in the report are stored with `put()` and written by `finish()`
to output/results/<step>.tex as LaTeX macros (\\newcommand), as in the course
labs, and to <step>.json. The report reads these files, so no number in the
text is typed by hand.
"""
from pathlib import Path
import json
import re
import sys
import time

import matplotlib
if "ipykernel" not in sys.modules:          # scripts save plots; the notebook also shows them
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATH_RAW = ROOT / "data" / "raw"
PATH_PANEL = ROOT / "data" / "panel.csv"
PATH_RESULTS = ROOT / "output" / "results"
PATH_PLOTS = ROOT / "output" / "analysis_plots"     # quick plots; report figures are in report_code/
for _p in (PATH_RAW, PATH_RESULTS, PATH_PLOTS):
    _p.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save_plot(fig, name):
    """Save a quick plot (as in the labs) and show it when run in the notebook."""
    fig.tight_layout()
    fig.savefig(PATH_PLOTS / name, dpi=150)
    if "ipykernel" in sys.modules:
        plt.show()
    plt.close(fig)

# The 21 EU member states with a national statutory minimum wage over the
# sample. Not included: AT, DK, FI, IT, SE (no statutory minimum wage) and CY
# (only since 2023). Germany enters in 2015, when its minimum wage started.
COUNTRIES = "BE BG CZ DE EE EL ES FR HR HU IE LT LU LV MT NL PL PT RO SI SK".split()

# Income years 2009-2024, that is EU-SILC waves 2010-2025 (see step 2).
FIRST_YEAR, LAST_YEAR = 2009, 2024

CONTROLS = ["unemp", "log_gdp_pc", "parttime"]
SEED = 20261109
N_BOOT = 9999


# --- data -------------------------------------------------------------------

def load_panel():
    return pd.read_csv(PATH_PANEL)


def sample(df, cols):
    """Rows with no missing value in the variables a regression uses."""
    return df.dropna(subset=list(cols)).copy()


def twfe(y, x, controls=True, fe="geo"):
    """Formula for model (1): y on x, the controls, country (or series-segment)
    dummies and year dummies. C() creates the dummies and drops one of each."""
    xs = [x] if isinstance(x, str) else list(x)
    rhs = xs + (CONTROLS if controls else []) + [f"C({fe})", "C(year)"]
    return f"{y} ~ " + " + ".join(rhs)


# --- estimation and inference -------------------------------------------------

def fit(formula, data, weights=None):
    """OLS (or WLS) with standard errors clustered by country.

    CR1 small-sample correction, and t tests and intervals use the t
    distribution with G - 1 = 20 degrees of freedom, as recommended with few
    clusters (Cameron and Miller 2015)."""
    groups = data["geo"].astype("category").cat.codes.to_numpy()
    model = smf.ols(formula, data) if weights is None else smf.wls(formula, data, weights=weights)
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)


def wild_cluster_p(formula, data, var, weights=None, n_boot=N_BOOT, seed=SEED):
    """Wild cluster bootstrap p-value for H0: coefficient on `var` = 0.

    Restricted residuals, Webb six-point weights drawn once per country and the
    CR1 t statistic (Cameron, Gelbach and Miller 2008; Webb 2023). With 21
    clusters the usual cluster-robust t test can over-reject; this p-value does
    not rely on the large-G approximation."""
    model = smf.ols(formula, data) if weights is None else smf.wls(formula, data, weights=weights)
    X, y = np.asarray(model.wexog), np.asarray(model.wendog)   # weighted data if WLS
    j = model.exog_names.index(var)
    groups = data["geo"].astype("category").cat.codes.to_numpy()
    G = groups.max() + 1
    n, k = X.shape

    XtX_inv = np.linalg.pinv(X.T @ X)
    c = XtX_inv[j] @ X.T                                   # beta_j = c @ y
    M = np.eye(n) - X @ XtX_inv @ X.T                      # residual-maker matrix
    scale = G / (G - 1) * (n - 1) / (n - k)                # CR1 correction
    onehot = np.zeros((n, G))
    onehot[np.arange(n), groups] = 1.0

    def t_stats(Y):                                        # Y has shape (draws, n)
        beta = Y @ c
        S = ((Y @ M.T) * c) @ onehot                       # cluster scores
        return beta / np.sqrt(scale * (S ** 2).sum(axis=1))

    t_obs = t_stats(y[None, :])[0]
    Xr = np.delete(X, j, axis=1)                           # model with H0 imposed
    fitted_r = Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    resid_r = y - fitted_r
    rng = np.random.default_rng(seed)
    webb = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    t_boot = np.empty(n_boot)
    for start in range(0, n_boot, 2000):                   # chunks keep memory small
        b = min(2000, n_boot - start)
        W = webb[rng.integers(0, 6, size=(b, G))][:, groups]
        t_boot[start:start + b] = t_stats(fitted_r[None, :] + W * resid_r[None, :])
    return float(np.mean(np.abs(t_boot) >= abs(t_obs)))


def coef(res, var, p_boot=None, scale=1.0):
    """Estimate, SE, 95% interval and p-values of one coefficient, optionally
    rescaled (scale=0.1 turns the log real minimum wage into 'per 10%')."""
    lo, hi = res.conf_int().loc[var]
    return {"b": res.params[var] * scale, "se": res.bse[var] * scale,
            "ci_lo": lo * scale, "ci_hi": hi * scale, "p": float(res.pvalues[var]),
            "p_boot": p_boot, "n": int(res.nobs),
            "countries": int(res.model.data.frame["geo"].nunique())}


def show(rows):
    """Print a dict of coef() results as a small table."""
    t = pd.DataFrame(rows).T
    cols = [c for c in ["b", "se", "ci_lo", "ci_hi", "p", "p_boot", "n"] if c in t]
    print(t[cols].astype(float).round(3).to_string())


def mde(se, df=len(COUNTRIES) - 1, power=0.8, alpha=0.05):
    """Minimum detectable effect: smallest true effect a two-sided 5% test
    detects with 80% probability, (t_{0.975} + t_{0.80}) x SE."""
    return (stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * se


def within_r2(res, formula, data):
    """Share of the variation left after the country and year dummies that the
    regressors explain."""
    fe = [t for t in ["C(geo)", "C(year)", "C(segment)"] if t in formula]
    y = formula.split("~")[0].strip()
    ssr_fe = smf.ols(f"{y} ~ " + " + ".join(fe), data=data).fit().ssr
    return 1 - res.ssr / ssr_fe


# --- saving results (same idea as put()/finish() in the course labs) --------------

class Results:
    """Collect numbers for the report. put() stores a formatted LaTeX macro,
    store() keeps structured results for the tables and figures."""

    def __init__(self, step):
        self.step, self.macros, self.data = step, {}, {}

    def put(self, name, value, digits=2, sign=False, pct=False):
        if not re.fullmatch(r"[A-Za-z]+", name):
            raise ValueError(f"LaTeX macro names need letters only: {name}")
        v = float(value)
        if digits == 0:
            txt = f"{v:,.0f}".replace(",", "{,}")
        else:
            txt = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
        self.macros[name] = f"\\ensuremath{{{txt}}}" + ("\\%" if pct else "")

    def put_text(self, name, text):
        self.macros[name] = text

    def store(self, key, value):
        self.data[key] = value

    def finish(self):
        tex = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in self.macros.items())
        _write(PATH_RESULTS / f"{self.step}.tex", tex)
        _write(PATH_RESULTS / f"{self.step}.json", json.dumps(self.data, indent=2, default=_json))
        print(f"Saved {len(self.macros)} numbers and {len(self.data)} result blocks to "
              f"output/results/{self.step}.tex/.json")


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    raise TypeError(type(x))


def _write(path, text):
    """Write a file, retrying briefly: OneDrive can lock a file while it syncs."""
    for attempt in range(5):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(2)
