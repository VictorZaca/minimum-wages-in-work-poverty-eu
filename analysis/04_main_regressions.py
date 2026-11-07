"""Step 4: main regressions. Paths are relative to this file.

Model (1) of the report, for country i and income year t:

    IWP_it = beta * MW_it + gamma' X_it + alpha_i + delta_t + u_it

IWP is in-work poverty among employees (percentage points). MW is the log real
minimum wage or the Kaitz index. X holds the unemployment rate, log real GDP
per capita and the part-time share. alpha_i are country dummies and delta_t
year dummies. Standard errors are clustered by country; every coefficient of
interest also gets a wild cluster bootstrap p-value (p_boot).
"""
import numpy as np
import pandas as pd

from tools import (CONTROLS, Results, coef, fit, load_panel, mde, sample, show, twfe,
                   wild_cluster_p, within_r2)

R = Results("step4")
df = load_panel()
base = ["iwp_sal", "log_real_mw"] + CONTROLS
d_mw = sample(df, base)
d_k = sample(df, base + ["kaitz"])
d_all = sample(df, base + ["iwp"])
d_all_k = sample(df, base + ["iwp", "kaitz"])

# %% 4(a) From pooled OLS to two-way fixed effects
# Column (1) compares countries with each other. Rich countries have higher
# minimum wages and lower poverty for many other reasons (benefits, family
# structure), so the pooled slope suffers from omitted variable bias.
# Country dummies (2) keep only changes within a country; year dummies (3)
# remove shocks common to all countries, such as COVID or the 2022 energy
# prices. Column (4) is our main specification.
SPECS = {
    "(1)": ("iwp_sal ~ log_real_mw", d_mw),
    "(2)": ("iwp_sal ~ log_real_mw + " + " + ".join(CONTROLS) + " + C(geo)", d_mw),
    "(3)": (twfe("iwp_sal", "log_real_mw", controls=False), d_mw),
    "(4)": (twfe("iwp_sal", "log_real_mw"), d_mw),
    "(5)": (twfe("iwp_sal", "kaitz"), d_k),
    "(6)": (twfe("iwp_sal", ["log_real_mw", "kaitz"]), d_k),
    "(7)": (twfe("iwp", "log_real_mw"), d_all),
    "(8)": (twfe("iwp", "kaitz"), d_all_k),
}
results, fits = {}, {}
for key, (formula, data) in SPECS.items():
    res = fit(formula, data)
    fits[key] = res
    entry = {"formula": formula, "n": int(res.nobs), "within_r2": within_r2(res, formula, data) if "C(" in formula else None}
    for v in ["log_real_mw", "kaitz"]:
        if v in res.params:
            entry[v] = coef(res, v, wild_cluster_p(formula, data, v))
    year_terms = [t for t in res.params.index if t.startswith("C(year)")]
    if year_terms:
        entry["F_year_p"] = float(res.f_test(" = 0, ".join(year_terms) + " = 0").pvalue)
    ctrl = [c for c in CONTROLS if c in res.params]
    if ctrl:
        entry["F_controls_p"] = float(res.f_test(", ".join(f"{c} = 0" for c in ctrl)).pvalue)
    results[key] = entry
show({f"{k} {v}": results[k][v] for k in results for v in ["log_real_mw", "kaitz"] if v in results[k]})
print(pd.DataFrame({k: {"within R2": e["within_r2"], "F year p": e.get("F_year_p"),
                        "F controls p": e.get("F_controls_p")} for k, e in results.items()}).round(3).to_string())
r4 = fits["(4)"]
keep = ["log_real_mw"] + CONTROLS
print("Column (4), slopes (country and year dummies not shown):")
print(pd.DataFrame({"coef": r4.params[keep], "cluster SE": r4.bse[keep], "t": r4.tvalues[keep],
                    "p (t, 20 df)": r4.pvalues[keep]}).round(3).to_string())
R.store("ladder", results)
# Interpretation:
# Every estimate for employees is small relative to its standard error and
# none is significant. The pooled slope and the fixed effects slopes have the
# same sign but different sizes, as expected if countries differ in ways the
# pooled model ignores. The year dummies are jointly significant at 10%
# without controls (column 3) but not once the controls are added (column 4).

# %% 4(b) Reading the main coefficient (column 4)
# Level-log form: a 10% higher real minimum wage changes in-work poverty by
# about beta/10 percentage points. We also report the 95% interval, the
# minimum detectable effect (80% power, 5% test) and the within R^2.
m = results["(4)"]["log_real_mw"]
per10 = {k: m[k] * 0.1 for k in ["b", "se", "ci_lo", "ci_hi"]}
mean_iwp = d_mw.iwp_sal.mean()
print("Coefficient:", round(m["b"], 3), "SE", round(m["se"], 3), "p", round(m["p"], 3), "p_boot", round(m["p_boot"], 3))
print("Per +10%:", {k: round(v, 3) for k, v in per10.items()})
print("Lower bound as % of mean in-work poverty:", round(100 * per10["ci_lo"] / mean_iwp, 1))
print("Minimum detectable effect per +10%:", round(mde(per10["se"]), 3))
R.put("MainB", m["b"], 2); R.put("MainSe", m["se"], 2); R.put("MainPboot", m["p_boot"], 2)
R.put("MainTen", per10["b"], 2); R.put("MainTenLo", per10["ci_lo"], 2); R.put("MainTenHi", per10["ci_hi"], 2, sign=True)
R.put("MainLoPctMean", -100 * per10["ci_lo"] / mean_iwp, 0)
R.put("MainMde", mde(per10["se"]), 2)
R.put("MainWithinR", results["(4)"]["within_r2"], 3)
R.put("MainFControlsP", results["(4)"]["F_controls_p"], 2)
R.put("MainFYearP", results["(4)"]["F_year_p"], 2)
R.put("NoCtrlFYearP", results["(3)"]["F_year_p"], 2)
# Interpretation:
# The point estimate is close to zero and the 95% interval excludes large
# reductions: at its lower end, a 10% higher minimum wage would lower
# in-work poverty by less than 5% of its mean. The regressors explain almost
# none of the within-country variation, and the controls are jointly
# insignificant.

# %% 4(c) The relative minimum wage: Kaitz index
# A relative poverty line may respond to the minimum wage relative to other
# wages rather than to its purchasing power, so we also use the Kaitz index
# (minimum wage as % of the mean wage), alone (5) and with the real wage (6).
k5 = results["(5)"]["kaitz"]
print("Kaitz (5):", {x: round(k5[x], 3) for x in ["b", "se", "ci_lo", "ci_hi", "p", "p_boot"]})
print("Both (6):", {v: round(results["(6)"][v]["b"], 3) for v in ["log_real_mw", "kaitz"]},
      {v: round(results["(6)"][v]["p_boot"], 2) for v in ["log_real_mw", "kaitz"]})
R.put("KaitzB", k5["b"], 2); R.put("KaitzLo", k5["ci_lo"], 2); R.put("KaitzHi", k5["ci_hi"], 2, sign=True)
R.put("KaitzPboot", k5["p_boot"], 2); R.put("KaitzMde", mde(k5["se"]), 2)
# Interpretation:
# One more point of Kaitz index goes with 0.05 points less in-work poverty,
# not significant, and adding both measures together changes nothing.

# %% 4(d) All employed people and the role of Romania
# For all employed people (self-employed included) the estimates are more
# negative. We re-estimate the four headline models without Romania, the
# country where in-work poverty fell mostly outside wage employment (step 3).
no_ro = {}
for key in ["(4)", "(5)", "(7)", "(8)"]:
    formula, data = SPECS[key]
    v = "kaitz" if key in ("(5)", "(8)") else "log_real_mw"
    dr = data[data.geo != "RO"]
    no_ro[key] = coef(fit(formula, dr), v, wild_cluster_p(formula, dr, v))
show({k + " " + ("with RO"): results[k]["kaitz" if k in ("(5)", "(8)") else "log_real_mw"] for k in no_ro})
show({k + " without RO": e for k, e in no_ro.items()})
R.store("no_romania", no_ro)
a7, a8 = results["(7)"]["log_real_mw"], results["(8)"]["kaitz"]
R.put("AllTen", a7["b"] * 0.1, 2); R.put("AllTenNoRo", no_ro["(7)"]["b"] * 0.1, 2, sign=True)
R.put("AllKaitz", a8["b"], 2); R.put("AllKaitzP", a8["p"], 2); R.put("AllKaitzPboot", a8["p_boot"], 2)
R.put("AllKaitzNoRo", no_ro["(8)"]["b"], 2)
R.put("MainTenNoRo", no_ro["(4)"]["b"] * 0.1, 2, sign=True)
# Interpretation:
# Without Romania the estimates for all employed people move to about zero,
# while those for employees barely change. The more negative all-employed
# estimates reflect the fall of poverty among Romania's self-employed.

# %% 4(e) Comparison with the simulations behind the Directive
# The JRC microsimulation (Gruenberger et al. 2021) predicts that raising every
# statutory minimum wage to 50% of the gross average wage would cut in-work
# poverty by 12-13% on average. What do our estimates imply for the same move?
# We take the 2019 Kaitz index (MW / mean wage), compute the rise needed to
# reach 50% in each country, and apply the main coefficients (columns 4 and 5).
k19 = df[df.year == 2019].set_index("geo")[["kaitz", "iwp_sal"]].dropna()
k19["dlog"] = np.log(50 / k19.kaitz).clip(lower=0)            # needed log increase
k19["dpts"] = (50 - k19.kaitz).clip(lower=0)                  # needed Kaitz points
below = k19[k19.kaitz < 50]
print(f"{len(below)} of {len(k19)} countries below 50% in 2019; mean needed increase "
      f"{100 * (np.exp(below.dlog) - 1).mean():.1f}%")
sim = {}
for name, key, var, col in [("real minimum wage", "(4)", "log_real_mw", "dlog"), ("Kaitz index", "(5)", "kaitz", "dpts")]:
    e = results[key][var]
    rel = {b: 100 * (e[b] * k19[col] / k19.iwp_sal).mean() for b in ["b", "ci_lo", "ci_hi"]}
    sim[name] = rel
    print(f"Implied average change in in-work poverty using the {name}: {rel['b']:.1f}% "
          f"(95% CI {rel['ci_lo']:.1f}% to {rel['ci_hi']:.1f}%)")
R.store("jrc_comparison", {"countries_below": len(below), "countries": len(k19),
                           "mean_needed_increase": 100 * (np.exp(below.dlog) - 1).mean(), "implied": sim})
R.put("SimBelow", len(below), 0); R.put("SimCountries", len(k19), 0)
R.put("SimIncrease", 100 * (np.exp(below.dlog) - 1).mean(), 0)
R.put("SimMw", -sim["real minimum wage"]["b"], 1)
R.put("SimMwLo", -sim["real minimum wage"]["ci_lo"], 1)
R.put("SimMwHi", sim["real minimum wage"]["ci_hi"], 1)
R.put("SimK", -sim["Kaitz index"]["b"], 1)
R.put("SimKLo", -sim["Kaitz index"]["ci_lo"], 1)
R.put("SimKHi", sim["Kaitz index"]["ci_hi"], 1)
R.finish()
# Interpretation:
# Reaching 50% of the average wage would have required raising minimum wages
# by about 16% on average in 2019 (among the countries below 50%). With the real minimum wage coefficient this
# implies an average fall in in-work poverty of well under 1%, and the 95%
# interval excludes the 12-13% of the simulations. With the Kaitz index the
# implied fall is about 5%, and the simulated value sits at the edge of the
# interval. The simulations hold behaviour and other policies fixed; our
# estimates include whatever adjustment happened in practice.
