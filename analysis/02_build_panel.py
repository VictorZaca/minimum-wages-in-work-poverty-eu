"""Step 2: build the country-year panel data/panel.csv from data/raw/. Paths are relative to this file.

One row per country and income year, 2009-2024. The cells below show how we
date the EU-SILC observations and build the real minimum wage, plus a price
index for low-income households that we use as a check.
"""
import numpy as np
import pandas as pd

from tools import COUNTRIES, FIRST_YEAR, LAST_YEAR, PATH_PANEL, PATH_RAW, CONTROLS, Results

R = Results("step2")


def read(name):
    df = pd.read_csv(PATH_RAW / f"{name}.csv", dtype={"time": str})
    df["flag"] = df["flag"].fillna("")
    return df[df.value.notna()]


def by_year(name, col, lag=0, **select):
    """One series as (geo, year, col). lag=1 moves an EU-SILC wave to its income year."""
    df = read(name)
    for k, v in select.items():
        df = df[df[k] == v]
    df = df.assign(year=df.time.astype(int) - lag)
    assert not df.duplicated(["geo", "year"]).any(), name
    return df[["geo", "year", "value"]].rename(columns={"value": col})


# %% 2(a) Timing: date EU-SILC by income year
# EU-SILC wave w asks about income in calendar year w-1 (in all countries since
# Ireland aligned in 2020). A poverty rate from wave w therefore describes year
# w-1, and we match it with the minimum wage of w-1. Material deprivation
# describes conditions at the interview, so it keeps the survey year.
iw = read("inwork_poverty_employees").assign(year=lambda d: d.time.astype(int) - 1)
iw["brk"] = iw.flag.str.contains("b").astype(int)
outcomes = [
    iw[["geo", "year", "value", "brk"]].rename(columns={"value": "iwp_sal"}),
    by_year("inwork_poverty", "iwp", lag=1),
    by_year("inwork_poverty_worktime", "iwp_ft", lag=1, worktime="FT"),
    by_year("inwork_poverty_worktime", "iwp_pt", lag=1, worktime="PT"),
    by_year("inwork_poverty_household", "iwp_single", lag=1, hhcomp="A1"),
    by_year("inwork_poverty_household", "iwp_single_kids", lag=1, hhcomp="A1_DCH"),
    by_year("inwork_poverty_household", "iwp_couple_nokids", lag=1, hhcomp="A_GE2_NDCH"),
    by_year("inwork_poverty_household", "iwp_couple_kids", lag=1, hhcomp="A_GE2_DCH"),
    by_year("poverty_by_activity", "pov_sal_40", lag=1, wstatus="SAL", rskpovth="B_40"),
    by_year("poverty_by_activity", "pov_sal_50", lag=1, wstatus="SAL", rskpovth="B_50"),
    by_year("poverty_by_activity", "pov_sal_60", lag=1, wstatus="SAL", rskpovth="B_60"),
    by_year("poverty_by_activity", "pov_sal_70", lag=1, wstatus="SAL", rskpovth="B_70"),
    by_year("poverty_by_activity", "pov_retired", lag=1, wstatus="RET", rskpovth="B_60"),
    by_year("poverty_by_activity", "pov_unemployed", lag=1, wstatus="UNE", rskpovth="B_60"),
    by_year("poverty_threshold", "threshold_nac", lag=1),
    by_year("poverty_anchored_2008", "pov_anchored", lag=1),
    by_year("deprivation_q1", "depriv_q1", lag=0),              # survey-year dating
]
print("Employee in-work poverty: survey waves", iw.time.min(), "to", iw.time.max(),
      "-> income years", iw.year.min(), "to", iw.year.max())
print("Breaks flagged by Eurostat:", int(iw[iw.year.between(FIRST_YEAR, LAST_YEAR)].brk.sum()),
      "in", iw[iw.year.between(FIRST_YEAR, LAST_YEAR) & (iw.brk == 1)].geo.nunique(), "countries")

# %% 2(b) Real minimum wage
# Gross monthly minimum wage in national currency, averaged over the January
# and July values, divided by the country's own HICP (2015 = 100). We express
# it in 2015 purchasing power standards (PPS) so that levels are comparable
# across countries; within a country this is a constant factor, so it does not
# change any estimate that has country dummies.
mw = read("minwage_nac").assign(year=lambda d: d.time.str[:4].astype(int))
mw = mw.groupby(["geo", "year"], as_index=False)["value"].mean().rename(columns={"value": "mw_nac"})
pps2015 = read("minwage_pps").query("time.str.startswith('2015')", engine="python").groupby("geo")["value"].mean()

grid = pd.MultiIndex.from_product([COUNTRIES, range(FIRST_YEAR - 2, LAST_YEAR + 3)],
                                  names=["geo", "year"]).to_frame(index=False)
panel = grid
for part in [mw, by_year("hicp_index", "hicp"), by_year("kaitz_mean", "kaitz"),
             by_year("kaitz_median", "kaitz_median"), by_year("unemployment", "unemp"),
             by_year("gdp_pc_real", "gdp_pc"), by_year("parttime_share", "parttime"),
             by_year("employees_thousands", "employees"),
             by_year("employment_rate_low_educ", "emp_low_educ")] + outcomes:
    panel = panel.merge(part, how="left", on=["geo", "year"])

panel["real_mw"] = panel.mw_nac / panel.hicp * 100
base = panel[panel.year == 2015].set_index("geo")
panel["real_mw_pps"] = panel.real_mw * panel.geo.map(pps2015 / base.real_mw)
panel["log_real_mw"] = np.log(panel.real_mw_pps)
panel["log_gdp_pc"] = np.log(panel.gdp_pc)
# The Kaitz index is MW / mean wage, so the mean wage it implies is MW / Kaitz.
# We use it to split the Kaitz index into its numerator and denominator.
panel["log_real_mean_wage"] = np.log(panel.mw_nac / (panel.kaitz / 100) / panel.hicp * 100)
panel["log_real_threshold"] = np.log(panel.threshold_nac / panel.hicp * 100)
chk = panel[panel.year.between(FIRST_YEAR, LAST_YEAR)]
print(chk.groupby("geo").real_mw_pps.agg(["first", "last"]).round(0).T.to_string())

# %% 2(c) A price index for low-income households
# Food and energy drove the 2022-23 inflation, and they weigh more in the
# budgets of poor households. We build a fixed-weight (Laspeyres) index of the
# 12 HICP divisions with the budget shares of the poorest income quintile from
# each country's latest Household Budget Survey wave, and the same for the
# richest quintile for comparison.
div = read("hicp_divisions").assign(year=lambda d: d.time.astype(int))
hbs = read("hbs_budget_shares").assign(year=lambda d: d.time.astype(int))
wave = hbs.groupby("geo").year.max()
hbs = hbs[hbs.year == hbs.geo.map(wave)]
weights = hbs.pivot_table(index=["geo", "quant_inc"], columns="coicop", values="value")


def basket(q):
    w = weights.xs(q, level="quant_inc").stack().rename("w").reset_index()
    d = div.merge(w, on=["geo", "coicop"]).assign(wp=lambda x: x.w * x.value)
    g = d.groupby(["geo", "year"]).agg(wp=("wp", "sum"), w=("w", "sum"))
    return (g.wp / g.w).rename(f"p_{q.lower()}").reset_index()


panel = panel.merge(basket("QU1"), how="left", on=["geo", "year"]).merge(basket("QU5"), how="left", on=["geo", "year"])
base = panel[panel.year == 2015].set_index("geo")
for q in ["p_qu1", "p_qu5"]:
    panel[q] = panel[q] / panel.geo.map(base[q]) * 100            # 2015 = 100
panel["log_real_mw_q1"] = np.log(panel.mw_nac / panel.p_qu1 * 100 * panel.geo.map(pps2015 / base.mw_nac))
print("HBS wave used for the budget shares:", wave.value_counts().to_dict())
print("Food and housing/energy shares (per mille), poorest vs richest quintile, median country:")
print(weights[["CP01", "CP04"]].groupby(level="quant_inc").median().round(0).to_string())

# %% 2(d) Leads, lags, series segments
# Leads and lags of the minimum wage for the distributed-lag model (Lecture 7).
# A new series segment starts at each Eurostat break in the employee series,
# so we can give each segment its own level as a robustness check.
panel = panel.sort_values(["geo", "year"]).reset_index(drop=True)
for v in ["log_real_mw", "kaitz"]:
    g = panel.groupby("geo")[v]
    panel[f"{v}_lead1"], panel[f"{v}_lag1"] = g.shift(-1), g.shift(1)
panel["brk"] = panel.brk.fillna(0).astype(int)
panel["segment"] = panel.geo + "_" + panel.groupby("geo")["brk"].cumsum().astype(str)

panel = panel[panel.year.between(FIRST_YEAR, LAST_YEAR)].reset_index(drop=True)
assert not panel.duplicated(["geo", "year"]).any()
assert (panel.real_mw_pps.dropna() > 0).all()
panel.to_csv(PATH_PANEL, index=False)

main = panel.dropna(subset=["iwp_sal", "log_real_mw"] + CONTROLS)
print(f"Panel: {len(panel)} country-years. Main sample: {len(main)} observations, "
      f"{main.geo.nunique()} countries, income years {main.year.min()}-{main.year.max()}")
print("Missing in the main sample: Germany before 2015 (no minimum wage).",
      sorted(set(panel.geo + panel.year.astype(str)) - set(main.geo + main.year.astype(str))))
print("Observations per variable:\n" + panel.notna().sum().to_string())

R.put("NObs", len(main), 0)
R.put("NCountries", main.geo.nunique(), 0)
R.put("NBreaks", int(main.brk.sum()), 0)
R.put("NBreakCountries", main[main.brk == 1].geo.nunique(), 0)
R.put("NKaitz", main.kaitz.notna().sum(), 0)
R.put("NKaitzMedianCountries", main.dropna(subset=["kaitz_median"]).geo.nunique(), 0)
R.finish()
# Interpretation:
# The panel has 336 country-years and the main regressions use 330: the six
# German years before the minimum wage started in 2015 have no minimum wage.
# The Kaitz index is missing for some country-years, so Kaitz regressions use
# fewer observations. Poor households spend a larger share of their budget on
# food than rich ones, which is why we also deflate by their own price index.
