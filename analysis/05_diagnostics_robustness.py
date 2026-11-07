"""Step 5: residual diagnostics and robustness checks. Paths are relative to this file.

Each check changes one thing in the main model (column 4 or 5 of step 4) and
reports the change in employee in-work poverty for a 10% higher real minimum
wage and for a one-point higher Kaitz index. The lecture each idea comes from
is given in brackets.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
import matplotlib.pyplot as plt

from tools import (CONTROLS, Results, coef, fit, load_panel, sample, save_plot, show, twfe,
                   wild_cluster_p)

R = Results("step5")
df = load_panel()
Y = "iwp_sal"
d_mw = sample(df, [Y, "log_real_mw"] + CONTROLS)
d_k = sample(df, [Y, "kaitz"] + CONTROLS)


def both(f_mw, data_mw, f_k=None, data_k=None, v_mw="log_real_mw", v_k="kaitz", w_mw=None, w_k=None):
    """Estimate one check for the real minimum wage (per +10%) and the Kaitz index."""
    out = {}
    if f_mw is not None:
        out["mw"] = coef(fit(f_mw, data_mw, w_mw), v_mw, wild_cluster_p(f_mw, data_mw, v_mw, w_mw), 0.1)
    if f_k is not None:
        out["k"] = coef(fit(f_k, data_k, w_k), v_k, wild_cluster_p(f_k, data_k, v_k, w_k))
    return out


# %% 5(a) Heteroskedasticity (Lecture 6)
# Breusch-Pagan: regress the squared OLS residuals on all regressors; the
# special White test uses the fitted values and their square. Under
# homoskedasticity n*R^2 is chi-squared.
plain = smf.ols(twfe(Y, "log_real_mw"), d_mw).fit()
lm_bp, p_bp, _, _ = het_breuschpagan(plain.resid, plain.model.exog)
df_bp = np.linalg.matrix_rank(plain.model.exog) - 1
aux = smf.ols("u2 ~ yhat + I(yhat ** 2)", pd.DataFrame({"u2": plain.resid ** 2, "yhat": plain.fittedvalues})).fit()
lm_w = len(d_mw) * aux.rsquared
p_w = stats.chi2.sf(lm_w, 2)
print(f"Breusch-Pagan LM = {lm_bp:.1f} ({df_bp} df), p = {p_bp:.2g}")
print(f"Special White LM = {lm_w:.1f} (2 df), p = {p_w:.2g}")
sd_by_country = plain.resid.groupby(d_mw.geo).std().sort_values()
print("Residual SD by country:\n", sd_by_country.round(2).to_string())
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.scatter(plain.fittedvalues, plain.resid, s=10, alpha=0.5)
ax.axhline(0, color="black", lw=0.7)
ax.set(xlabel="Fitted in-work poverty (pp)", ylabel="Residual (pp)")
save_plot(fig, "step5_residuals_vs_fitted.png")
R.put("BpLm", lm_bp, 1); R.put("BpDf", df_bp, 0); R.put("WhiteLm", lm_w, 1)
R.put("ResSdMin", sd_by_country.min(), 2); R.put("ResSdMax", sd_by_country.max(), 1)
R.store("heteroskedasticity", {"bp": {"LM": lm_bp, "df": int(df_bp), "p": p_bp}, "white": {"LM": lm_w, "p": p_w},
                               "resid_sd": sd_by_country.to_dict()})
# Interpretation:
# Both tests reject constant variance. The residual spread differs a lot by
# country: the poverty series of some countries are much noisier than others.
# OLS is still unbiased, but its usual standard errors are wrong, so
# we use cluster-robust standard errors, which are also robust to
# heteroskedasticity. Step 5(c) tries WLS and FGLS.

# %% 5(b) Serial correlation (Lecture 7)
# Regress each residual on the previous year's residual of the same country.
u = d_mw.assign(u=plain.resid).sort_values(["geo", "year"])
u["u_lag"] = u.groupby("geo").u.shift(1)
ar = fit("u ~ u_lag - 1", u.dropna(subset=["u_lag"]))
print(f"Residual AR(1): rho = {ar.params.u_lag:.2f} (SE {ar.bse.u_lag:.2f})")
R.put("ArRho", ar.params.u_lag, 2)
R.store("ar1", {"rho": ar.params.u_lag, "se": ar.bse.u_lag})
# Interpretation:
# The residuals are strongly positively autocorrelated: a country with
# unexpectedly high poverty one year tends to have it the next year too.
# Standard errors that assume independent observations would be far too
# small. Clustering by country allows any correlation within a country.

# %% 5(c) Weighted least squares and FGLS (Lecture 6)
# Two ways to reweight. WLS by the number of employees gives the effect for
# the average EU employee instead of the average country. FGLS weights each
# country by the inverse of its residual variance from 5(a), so noisy series
# count less. Standard errors stay cluster-robust in case the variance model
# is wrong.
weights = {}
for name, data, v in [("mw", d_mw, "log_real_mw"), ("k", d_k, "kaitz")]:
    resid = smf.ols(twfe(Y, v), data).fit().resid
    weights[name] = 1 / (resid ** 2).groupby(data.geo).transform("mean")
wls = both(twfe(Y, "log_real_mw"), d_mw, twfe(Y, "kaitz"), d_k, w_mw=d_mw.employees, w_k=d_k.employees)
fgls = both(twfe(Y, "log_real_mw"), d_mw, twfe(Y, "kaitz"), d_k, w_mw=weights["mw"], w_k=weights["k"])
show({"WLS (employees), MW": wls["mw"], "WLS (employees), Kaitz": wls["k"],
      "FGLS, MW": fgls["mw"], "FGLS, Kaitz": fgls["k"]})
share = (weights["mw"].groupby(d_mw.geo).sum() / weights["mw"].sum()).sort_values(ascending=False)
print("FGLS weight share by country (%):", (100 * share).round(1).head(6).to_dict())
loo_fgls = {}
for g in sorted(d_mw.geo.unique()):
    dd = d_mw[d_mw.geo != g]
    w = 1 / (smf.ols(twfe(Y, "log_real_mw"), dd).fit().resid ** 2).groupby(dd.geo).transform("mean")
    loo_fgls[g] = fit(twfe(Y, "log_real_mw"), dd, w).params["log_real_mw"] * 0.1
print("FGLS per +10%, leaving one country out:", {g: round(b, 3) for g, b in loo_fgls.items()})
R.put("FglsTen", fgls["mw"]["b"], 2); R.put("FglsTenP", fgls["mw"]["p"], 2); R.put("FglsTenPboot", fgls["mw"]["p_boot"], 2)
R.put("FglsK", fgls["k"]["b"], 3); R.put("FglsKSe", fgls["k"]["se"], 3); R.put("FglsKPboot", fgls["k"]["p_boot"], 2)
R.put("FglsTopShare", 100 * share.iloc[:4].sum(), 0)
R.put("FglsNoRo", loo_fgls["RO"], 2, sign=True); R.put("FglsNoIe", loo_fgls["IE"], 2, sign=True)
R.put("WlsTen", wls["mw"]["b"], 2); R.put("WlsK", wls["k"]["b"], 2); R.put("WlsKPboot", wls["k"]["p_boot"], 2)
R.store("weighted", {"wls": wls, "fgls": fgls, "fgls_share": share.to_dict(), "fgls_loo": loo_fgls})
# Interpretation:
# For the Kaitz index both weighted estimators give negative estimates with
# bootstrap p-values below 5%; FGLS has about the same point estimate as OLS
# with a much smaller standard error. For the real
# minimum wage the clustered p-value is below 5% but the bootstrap p-value is
# not: four countries get most of the weight, so in effect there are few
# clusters, and that is when the usual cluster-robust test rejects too often.
# Leaving out Romania or Ireland removes most of the FGLS estimate.

# %% 5(d) Robustness checks
# - no controls: the controls could themselves respond to the minimum wage
#   (bad controls)
# - a separate level after every EU-SILC break (series-segment dummies)
# - survey-year timing: the minimum wage of the interview year
# - dropping income years 2020-21 (COVID) and dropping Romania
# - deflating by the low-income price index; the Kaitz index on the median wage
checks = {}
checks["no_controls"] = both(twfe(Y, "log_real_mw", controls=False), d_mw, twfe(Y, "kaitz", controls=False), d_k)
checks["segments"] = both(twfe(Y, "log_real_mw", fe="segment"), d_mw, twfe(Y, "kaitz", fe="segment"), d_k)
checks["survey_year"] = both(twfe(Y, "log_real_mw_lead1"), sample(df, [Y, "log_real_mw_lead1"] + CONTROLS),
                             twfe(Y, "kaitz_lead1"), sample(df, [Y, "kaitz_lead1"] + CONTROLS),
                             v_mw="log_real_mw_lead1", v_k="kaitz_lead1")
covid = lambda d: d[~d.year.isin([2020, 2021])]
checks["no_covid"] = both(twfe(Y, "log_real_mw"), covid(d_mw), twfe(Y, "kaitz"), covid(d_k))
no_ro = lambda d: d[d.geo != "RO"]
checks["no_romania"] = both(twfe(Y, "log_real_mw"), no_ro(d_mw), twfe(Y, "kaitz"), no_ro(d_k))
checks["q1_deflator"] = both(twfe(Y, "log_real_mw_q1"), sample(df, [Y, "log_real_mw_q1"] + CONTROLS), v_mw="log_real_mw_q1")
checks["kaitz_median"] = both(None, None, twfe(Y, "kaitz_median"), sample(df, [Y, "kaitz_median"] + CONTROLS), v_k="kaitz_median")

# %% 5(e) First differences within series segments (Lecture 7)
# Poverty rates are persistent, so we also estimate the model in first
# differences, which removes the country dummies. We never difference across
# a series break.
fd = df.sort_values(["geo", "year"]).copy()
for v in [Y, "log_real_mw", "kaitz"] + CONTROLS:
    fd["d_" + v] = fd.groupby("geo")[v].diff()
fd = fd[(fd.groupby("geo").year.diff() == 1) & (fd.brk == 0)]
dctrl = " + ".join("d_" + c for c in CONTROLS)
need = [f"d_{Y}"] + ["d_" + c for c in CONTROLS]
checks["first_differences"] = both(f"d_{Y} ~ d_log_real_mw + {dctrl} + C(year)", sample(fd, need + ["d_log_real_mw"]),
                                   f"d_{Y} ~ d_kaitz + {dctrl} + C(year)", sample(fd, need + ["d_kaitz"]),
                                   v_mw="d_log_real_mw", v_k="d_kaitz")
checks["wls_employees"], checks["fgls"] = wls, fgls
show({f"{k}, {s}": e for k, c in checks.items() for s, e in c.items()})
R.store("checks", checks)
pb = [e["p_boot"] for k, c in checks.items() if k not in ("fgls", "wls_employees") for e in c.values()]
R.put("ChecksMinPboot", np.floor(100 * min(pb)) / 100, 2)   # rounded down, so "none below" is true
# Interpretation:
# No check moves the employee estimates far from zero, and none of these
# checks has a bootstrap p-value below 0.15. First differences give the same
# answer as the dummies.

# %% 5(f) Strict exogeneity: a distributed lag with a lead (Lecture 7)
# If governments raised the minimum wage in reaction to poverty, or in years
# when poverty was about to change, next year's minimum wage would "predict"
# this year's poverty. We add the lead and one lag of the minimum wage.
fdl = {}
for v, scale in [("log_real_mw", 0.1), ("kaitz", 1.0)]:
    d = sample(df, [Y, f"{v}_lead1", v, f"{v}_lag1"] + CONTROLS)
    f = twfe(Y, [f"{v}_lead1", v, f"{v}_lag1"])
    r = fit(f, d)
    cum = r.t_test(f"{v} + {v}_lag1 = 0")
    fdl[v] = {"lead": coef(r, f"{v}_lead1", wild_cluster_p(f, d, f"{v}_lead1"), scale),
              "current": coef(r, v, None, scale), "lag": coef(r, f"{v}_lag1", None, scale),
              "long_run": float(cum.effect.squeeze()) * scale, "long_run_p": float(cum.pvalue)}
show({f"{v}: {t}": fdl[v][t] for v in fdl for t in ["lead", "current", "lag"]})
print({v: (round(fdl[v]["long_run"], 3), round(fdl[v]["long_run_p"], 2)) for v in fdl})
R.store("distributed_lag", fdl)
R.put("LeadTen", fdl["log_real_mw"]["lead"]["b"], 2); R.put("LeadTenPboot", fdl["log_real_mw"]["lead"]["p_boot"], 2)
R.put("LeadK", fdl["kaitz"]["lead"]["b"], 2); R.put("LeadKPboot", fdl["kaitz"]["lead"]["p_boot"], 2)
R.put("LongRunTen", fdl["log_real_mw"]["long_run"], 2); R.put("LongRunTenP", fdl["log_real_mw"]["long_run_p"], 2)
# Interpretation:
# The lead is small and insignificant, so we see no sign of the feedback that
# would violate strict exogeneity. The test has little power, so it cannot
# prove strict exogeneity. The current and lagged effects add up to a small,
# insignificant positive number.

# %% 5(g) Functional form (Lecture 5)
# A log outcome turns the coefficient into an elasticity; a quadratic in the
# Kaitz index allows the effect to change with the level of the minimum wage.
r_log = fit(twfe("np.log(iwp_sal)", "log_real_mw"), d_mw)
r_q = fit(twfe(Y, ["kaitz", "I(kaitz ** 2)"]), d_k)
p_quad = float(r_q.f_test("kaitz = 0, I(kaitz ** 2) = 0").pvalue)
within_sd = lambda s, d: float((s - s.groupby(d.geo).transform("mean")).std())
r4, r5 = fit(twfe(Y, "log_real_mw"), d_mw), fit(twfe(Y, "kaitz"), d_k)
std_mw = r4.params["log_real_mw"] * within_sd(d_mw.log_real_mw, d_mw) / within_sd(d_mw[Y], d_mw)
std_k = r5.params["kaitz"] * within_sd(d_k.kaitz, d_k) / within_sd(d_k[Y], d_k)
print(f"Elasticity (log outcome): {r_log.params.log_real_mw:.3f} (SE {r_log.bse.log_real_mw:.3f})")
print(f"Quadratic Kaitz, joint F test p = {p_quad:.2f}")
print(f"Standardized (within-country SDs): real MW {std_mw:.3f}, Kaitz {std_k:.3f}")
R.put("Elast", r_log.params.log_real_mw, 2); R.put("ElastSe", r_log.bse.log_real_mw, 2)
R.put("QuadP", p_quad, 2); R.put("StdMw", std_mw, 2); R.put("StdK", std_k, 2)
R.store("functional_form", {"elasticity": coef(r_log, "log_real_mw"), "quad_F_p": p_quad,
                            "standardized": {"mw": std_mw, "k": std_k}})
# Interpretation:
# A 10% higher minimum wage goes with about 1% lower in-work poverty in the log
# model, not significant. The quadratic adds nothing. In standardized units a
# one-SD within-country change in the minimum wage moves poverty by a small
# fraction of its within-country SD.

# %% 5(h) Influential observations (Lecture 5)
# Cook's distance for single country-years, and the main estimate when one
# whole country is dropped at a time.
cook = d_mw.assign(cook=plain.get_influence().cooks_distance[0]).sort_values("cook", ascending=False)
print(cook[["geo", "year", "cook"]].head(5).to_string(index=False))
loo = []
for g in sorted(d_mw.geo.unique()):
    a = fit(twfe(Y, "log_real_mw"), d_mw[d_mw.geo != g])
    b = fit(twfe(Y, "kaitz"), d_k[d_k.geo != g])
    loo.append({"dropped": g, "mw": a.params.log_real_mw * 0.1, "mw_se": a.bse.log_real_mw * 0.1,
                "k": b.params.kaitz, "k_se": b.bse.kaitz})
loo = pd.DataFrame(loo)
print(loo.round(3).to_string(index=False))
R.put("CookMax", cook.cook.iloc[0], 2)
R.put("LooMin", loo.mw.min(), 2); R.put("LooMax", loo.mw.max(), 2, sign=True)
R.put("LooKMin", loo.k.min(), 2); R.put("LooKMax", loo.k.max(), 2)
R.store("leave_one_out", loo.to_dict("records"))
R.store("cook_top", cook[["geo", "year", "cook"]].head(5).to_dict("records"))
R.finish()
# Interpretation:
# No single country-year dominates the fit, and dropping any one country keeps
# the main estimate close to zero.
