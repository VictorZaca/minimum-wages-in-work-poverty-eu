"""Step 3: descriptive statistics and motivating evidence. Paths are relative to this file.

Before any regression we look at the data: how much the minimum wage moves
within countries, what happened in France and Germany, and whether countries
whose minimum wage grew more saw in-work poverty fall more.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from tools import CONTROLS, Results, load_panel, sample

R = Results("step3")
df = load_panel()
main = sample(df, ["iwp_sal", "log_real_mw"] + CONTROLS)

# %% 3(a) Summary statistics
# The "within SD" removes each country's mean. Only this variation is left once
# we add country dummies, so it tells us whether there is enough to learn from.
VARS = {"iwp_sal": "In-work poverty, employees (%)", "iwp": "In-work poverty, all employed (%)",
        "real_mw_pps": "Real minimum wage (2015 PPS/month)", "kaitz": "Kaitz index (MW / mean wage, %)",
        "unemp": "Unemployment rate (%)", "gdp_pc": "Real GDP per capita (2020 EUR, 000s)",
        "parttime": "Part-time share of employment (%)"}
rows = {}
for v, label in VARS.items():
    s = main[v] / (1000 if v == "gdp_pc" else 1)
    rows[label] = {"N": s.count(), "Mean": s.mean(), "SD": s.std(),
                   "Within SD": (s - s.groupby(main.geo).transform("mean")).std(), "Min": s.min(), "Max": s.max()}
table1 = pd.DataFrame(rows).T
print(table1.round(2).to_string())
within_mw = (main.log_real_mw - main.groupby("geo").log_real_mw.transform("mean")).std()
print("Within-country SD of the log real minimum wage:", round(within_mw, 3))
R.store("table1", table1.reset_index(names="variable").to_dict("records"))
R.put("WithinSdMw", 100 * within_mw, 0)
R.put("WithinSdMwLog", within_mw, 2)
R.put("MeanIwpSal", main.iwp_sal.mean(), 1)
R.put("MeanIwp", main.iwp.mean(), 1)
# Interpretation:
# In-work poverty among employees averages about 6% and the real minimum wage
# varies by about 19% around each country's own mean. That is a lot of
# within-country variation, so a null result later cannot be blamed on a
# minimum wage that never moved.

# %% 3(b) France and Germany, 2015-2024
# France indexes its minimum wage (SMIC) automatically to prices; Germany sets
# it by commission and raised it by law to EUR 12 per hour in October 2022.
fd = df[df.geo.isin(["FR", "DE"]) & df.year.between(2015, 2024)].copy()
fd["mw_index"] = fd.real_mw / fd.geo.map(fd[fd.year == 2015].set_index("geo").real_mw) * 100
wide = fd.pivot(index="year", columns="geo", values=["mw_index", "iwp_sal", "brk"])
print(wide.round(1).to_string())
fr, de = fd[fd.geo == "FR"].set_index("year"), fd[fd.geo == "DE"].set_index("year")
R.put("FrMwMaxDev", (fr.mw_index - 100).abs().max(), 1)
R.put("DeMwGrowth", de.loc[2024, "mw_index"] - 100, 0)
R.put("DeMwGrowthPreHike", de.loc[2021, "mw_index"] - 100, 0)
for g, s in [("Fr", fr), ("De", de)]:
    R.put(g + "IwpStart", s.loc[2015, "iwp_sal"], 1)
    R.put(g + "IwpEnd", s.loc[2024, "iwp_sal"], 1)
R.put("DeIwpNineteen", de.loc[2019, "iwp_sal"], 1)
R.put("FrIwpTwentyOne", fr.loc[2021, "iwp_sal"], 1)
R.put("FrIwpTwentyThree", fr.loc[2023, "iwp_sal"], 1)
R.put("DeIwpTwentyOne", de.loc[2021, "iwp_sal"], 1)
R.put_text("DeBreakYears", " and ".join(str(y) for y in de.index[de.brk == 1]))
es = df[df.geo == "ES"].set_index("year").real_mw
print("Spain, real minimum wage change 2018 to 2019 (%):", round(100 * (es[2019] / es[2018] - 1), 1))
R.put("EsMwJump", 100 * (es[2019] / es[2018] - 1), 0)
R.store("fr_de", fd[["geo", "year", "mw_index", "iwp_sal", "brk"]].to_dict("records"))
# Interpretation:
# France's real minimum wage stayed close to its 2015 value while Germany's
# rose, mostly after October 2022. Employee in-work poverty rose slightly in
# France and fell in Germany, but the German fall came in 2019-2021, before the
# big increase, and in years with breaks in the German series. Two countries
# cannot separate the minimum wage from everything else that changed.

# %% 3(c) Long differences across all 21 countries
# Change in in-work poverty against the change in the log real minimum wage
# between 2015 and 2024, one point per country. Robust (HC1) standard error.
y0, y1 = 2015, 2024
w = df[df.year.isin([y0, y1])].pivot(index="geo", columns="year", values=["iwp_sal", "log_real_mw"])
ld = pd.DataFrame({"d_iwp": w["iwp_sal"][y1] - w["iwp_sal"][y0],
                   "d_mw": 100 * (w["log_real_mw"][y1] - w["log_real_mw"][y0])}).dropna()
long_diff = smf.ols("d_iwp ~ d_mw", data=ld).fit(cov_type="HC1")
print(ld.sort_values("d_mw").round(2).T.to_string())
print(long_diff.summary().tables[1])
R.put("LdSlope", long_diff.params.d_mw, 3)
R.put("LdSe", long_diff.bse.d_mw, 3)
R.put("LdMwMax", ld.d_mw.max(), 0)
R.put_text("LdMwMaxCountry", {"RO": "Romania", "BG": "Bulgaria", "PL": "Poland"}.get(ld.d_mw.idxmax(), ld.d_mw.idxmax()))
R.put("LdMwMin", ld.d_mw.min(), 0)
R.store("long_diff", {"slope": long_diff.params.d_mw, "se": long_diff.bse.d_mw,
                      "intercept": long_diff.params.Intercept, "points": ld.reset_index().to_dict("records")})
# Interpretation:
# Countries whose real minimum wage grew more did not see in-work poverty fall
# more: the slope is about zero. This is a raw comparison with no controls,
# but it already points in the direction of the regression results.

# %% 3(d) Romania: employees versus all employed people
# The minimum wage covers employees but not the self-employed. In Romania the
# poverty rates of these two groups are much further apart than elsewhere.
ro = df[df.geo == "RO"].set_index("year")
gap = (df.iwp - df.iwp_sal).groupby(df.geo).mean()
print(ro.loc[[2015, 2024], ["iwp", "iwp_sal", "real_mw"]])
print("Average gap (all employed minus employees), by country:\n", gap.round(1).sort_values().to_string())
R.put("RoIwpStart", ro.loc[2015, "iwp"], 1)
R.put("RoIwpEnd", ro.loc[2024, "iwp"], 1)
R.put("RoSalStart", ro.loc[2015, "iwp_sal"], 1)
R.put("RoSalEnd", ro.loc[2024, "iwp_sal"], 1)
R.put("RoMwGrowth", 100 * (ro.loc[2024, "real_mw"] / ro.loc[2015, "real_mw"] - 1), 0)
R.put("RoGap", gap["RO"], 0)
R.put("OtherGap", gap.drop("RO").median(), 1)
# Interpretation:
# In Romania in-work poverty among all workers fell much more than among
# employees, so most of the fall happened outside wage employment. Romania's
# real minimum wage also grew the most, so regressions for all employed
# people may pick up this compositional change. This is why our main outcome
# is the employee rate.

# %% 3(e) Two facts about prices and the 2022-23 inflation
# How did real minimum wages change from 2021 to 2023? And did prices rise
# faster for the poorest fifth of households than for the richest fifth?
s = df[df.year.isin([2021, 2023])].pivot(index="geo", columns="year", values="real_mw").dropna()
surge = (100 * (s[2023] / s[2021] - 1)).sort_values()
print("Real minimum wage, % change 2021-2023:\n", surge.round(1).to_string())
c = df[df.year.isin([2020, 2024])].pivot(index="geo", columns="year", values=["p_qu1", "p_qu5"])
q_gap = 100 * (c["p_qu1"][2024] / c["p_qu1"][2020] - 1) - 100 * (c["p_qu5"][2024] / c["p_qu5"][2020] - 1)
print("Price rise 2020-2024, poorest minus richest quintile (points):\n", q_gap.round(1).sort_values().to_string())
R.put("SurgeFr", surge["FR"], 1, sign=True)
R.put("SurgeMin", surge.min(), 1, sign=True)
R.put("SurgeMax", surge.max(), 1, sign=True)
names = {"CZ": "Czechia", "DE": "Germany", "EE": "Estonia", "LV": "Latvia", "SK": "Slovakia", "HU": "Hungary"}
R.put_text("SurgeMinCountry", names.get(surge.idxmin(), surge.idxmin()))
R.put_text("SurgeMaxCountry", names.get(surge.idxmax(), surge.idxmax()))
R.put("BasketCountries", (q_gap > 0).sum(), 0)
R.put("BasketGapMedian", q_gap.median(), 1)
R.store("surge", surge.to_dict())
R.finish()
# Interpretation:
# Real minimum wages moved in very different directions during the inflation
# surge, which is useful variation for the panel. Prices also rose faster for
# the poorest fifth of households in most countries, so deflating by the
# general HICP may overstate their real minimum wage; step 5 checks this.
