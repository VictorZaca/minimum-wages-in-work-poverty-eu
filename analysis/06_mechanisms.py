"""Step 6: mechanisms. Why might a higher minimum wage do little for in-work poverty? Paths are relative to this file.

Same model and controls as the main specification, with other outcomes:
deeper and shallower poverty lines, working time and household type, the
poverty line itself, groups whose income does not depend on the minimum wage
(a placebo), employment of the low-educated (selection) and material
deprivation.
"""
import numpy as np
import pandas as pd

from tools import CONTROLS, Results, coef, fit, load_panel, sample, show, twfe, wild_cluster_p

R = Results("step6")
df = load_panel()


def run(y, scale_mw=0.1):
    """Main model for outcome y, with the real minimum wage and with the Kaitz index."""
    row = {}
    for v, scale in [("log_real_mw", scale_mw), ("kaitz", 1.0)]:
        d = sample(df, [y, v] + CONTROLS)
        f = twfe(y, v)
        row[v] = coef(fit(f, d), v, wild_cluster_p(f, d, v), scale)
        row[v]["mean"] = float(d[y].mean())
        row[v]["pct_of_mean"] = 100 * row[v]["b"] / row[v]["mean"]
    return row


out = {}

# %% 6(a) Deeper and shallower poverty lines
# A full-time minimum wage job keeps most workers above a line set at 40% of
# the median, so the minimum wage may matter more for deep poverty. Eurostat
# publishes poverty rates of employees (aged 18+) at 40, 50, 60 and 70% of the
# median from table ilc_li04, for income years 2014-2024 only.
for y in ["pov_sal_40", "pov_sal_50", "pov_sal_60", "pov_sal_70"]:
    out[y] = run(y)
show({f"{y}, {v}": out[y][v] for y in ["pov_sal_40", "pov_sal_50", "pov_sal_60", "pov_sal_70"]
      for v in ["log_real_mw", "kaitz"]})
print("As % of each mean (real MW +10%):",
      {y: round(out[y]["log_real_mw"]["pct_of_mean"], 1) for y in ["pov_sal_40", "pov_sal_50", "pov_sal_60", "pov_sal_70"]})
same = sample(df, ["iwp_sal", "pov_sal_60"])
print("Correlation of ilc_li04 (60%) with our main outcome ilc_iw01:", round(np.corrcoef(same.iwp_sal, same.pov_sal_60)[0, 1], 3))
for y, nm in [("pov_sal_40", "Forty"), ("pov_sal_50", "Fifty"), ("pov_sal_60", "Sixty"), ("pov_sal_70", "Seventy")]:
    e = out[y]["log_real_mw"]
    R.put("Line" + nm, e["b"], 2); R.put("Line" + nm + "Pct", -e["pct_of_mean"], 0)
    R.put("Line" + nm + "Pboot", e["p_boot"], 2); R.put("Line" + nm + "Mean", e["mean"], 1)
# Interpretation:
# Relative to how common each kind of poverty is, the association is largest
# for the deepest line: a 10% higher minimum wage goes with about 8% less
# poverty below 40% of the median, against about 2% at 60%. None of these is
# significant at 5% with the bootstrap, so we only read it as a pattern.

# %% 6(b) Working time and household type
# EU-SILC publishes these breakdowns only for all employed people.
groups = {"iwp_ft": "Full-time", "iwp_pt": "Part-time", "iwp_single": "Single adult",
          "iwp_single_kids": "Single adult with children", "iwp_couple_nokids": "2+ adults, no children",
          "iwp_couple_kids": "2+ adults with children"}
for y in groups:
    out[y] = run(y)
show({f"{groups[y]}, {v}": out[y][v] for y in groups for v in ["log_real_mw", "kaitz"]})
R.put("MeanFt", out["iwp_ft"]["log_real_mw"]["mean"], 0); R.put("MeanPt", out["iwp_pt"]["log_real_mw"]["mean"], 0)
R.put("SingleTen", out["iwp_single"]["log_real_mw"]["b"], 2)
R.put("SingleK", out["iwp_single"]["kaitz"]["b"], 2); R.put("SingleKPboot", out["iwp_single"]["kaitz"]["p_boot"], 2)
R.put("SingleKidsTen", out["iwp_single_kids"]["log_real_mw"]["b"], 2)
R.put("CoupleKidsTen", out["iwp_couple_kids"]["log_real_mw"]["b"], 2)
R.put("CoupleTen", out["iwp_couple_nokids"]["log_real_mw"]["b"], 2)
R.put("MeanSingleKids", out["iwp_single_kids"]["log_real_mw"]["mean"], 0)
# Interpretation:
# Part-time workers are poor far more often than full-time workers, but both
# estimates are imprecise. The largest associations are for single adults,
# whose household lives on one wage; the smallest are for couples.

# %% 6(c) Do single adults respond differently from couples with children?
# Stack the two outcomes and interact the minimum wage with a single-adult
# dummy. Country x group and year x group dummies, controls with group-specific
# slopes; the interaction is the difference in slopes (Lecture 5).
stack = []
for y, single in [("iwp_single", 1), ("iwp_couple_kids", 0)]:
    s = sample(df, [y, "log_real_mw"] + CONTROLS).assign(single=single)
    s["y"] = s[y]
    stack.append(s)
st = pd.concat(stack, ignore_index=True)
st["mw_x_single"] = st.log_real_mw * st.single
st["cell"] = st.geo + "_" + st.single.astype(str)                          # country x group
st["year_single"] = np.where((st.single == 1) & (st.year > st.year.min()),   # extra year effects
                             st.year.astype(str), "base")                    # for single adults
f = ("y ~ log_real_mw + mw_x_single + " + " + ".join(f"{c} + {c}:single" for c in CONTROLS)
     + " + C(cell) + C(year) + C(year_single)")
gap = coef(fit(f, st), "mw_x_single", wild_cluster_p(f, st, "mw_x_single"), 0.1)
print("Extra slope for single adults, per +10%:", {k: round(gap[k], 3) for k in ["b", "se", "p", "p_boot"]})
out["single_minus_couple_kids"] = gap
R.put("GapTen", gap["b"], 2); R.put("GapPboot", gap["p_boot"], 2)
# Interpretation:
# The difference has the expected sign but is not significant.

# %% 6(d) Does the poverty line move with the minimum wage?
# The at-risk-of-poverty line is 60% of the national median income. If the
# median rises with the minimum wage, part of any gain at the bottom is
# matched by a higher line. We regress the log real line on the log real
# minimum wage (an elasticity), and use the poverty rate against a line fixed
# at its 2008 real value, which cannot move.
out["log_real_threshold"] = run("log_real_threshold", scale_mw=1.0)
out["pov_anchored"] = run("pov_anchored")
show({"Log real threshold (elasticity), MW": out["log_real_threshold"]["log_real_mw"],
      "Anchored poverty, MW": out["pov_anchored"]["log_real_mw"]})
R.put("ThreshElast", out["log_real_threshold"]["log_real_mw"]["b"], 2)
R.put("ThreshElastSe", out["log_real_threshold"]["log_real_mw"]["se"], 2)
R.put("ThreshTen", 10 * out["log_real_threshold"]["log_real_mw"]["b"], 1)
R.put("AnchTen", out["pov_anchored"]["log_real_mw"]["b"], 2)
R.put("AnchPboot", out["pov_anchored"]["log_real_mw"]["p_boot"], 2)
# Interpretation:
# A 10% higher real minimum wage goes with a higher real poverty line, but
# the elasticity is imprecise. Poverty against the fixed 2008 line responds
# more than relative poverty, again not significantly.

# %% 6(e) Placebo: people whose income does not come from wages
# Retired people's incomes do not depend on the minimum wage. If minimum wage
# rises coincided with good years in general, poverty among the retired would
# fall too. We use the at-risk-of-poverty rate of retired and unemployed
# people (18+, ilc_li04).
out["pov_retired"] = run("pov_retired")
out["pov_unemployed"] = run("pov_unemployed")
show({"Retired, MW": out["pov_retired"]["log_real_mw"], "Unemployed, MW": out["pov_unemployed"]["log_real_mw"]})
R.put("RetTen", out["pov_retired"]["log_real_mw"]["b"], 2, sign=True)
R.put("RetSe", out["pov_retired"]["log_real_mw"]["se"], 2)
R.put("UneTen", out["pov_unemployed"]["log_real_mw"]["b"], 2, sign=True)
R.put("UneSe", out["pov_unemployed"]["log_real_mw"]["se"], 2)
# Interpretation:
# Neither rate moves significantly with the minimum wage, so the placebo shows no sign that
# minimum wage increases came with general improvements we fail to control.

# %% 6(f) Selection into employment
# In-work poverty is measured among people who have a job. If a higher minimum
# wage pushed low-paid workers out of work, the remaining employees would be
# better paid on average and the measured rate could fall for that reason
# alone. We check the employment rate of people aged 20-64 with at most lower
# secondary education, the group most exposed to job losses.
out["emp_low_educ"] = run("emp_low_educ")
show({"Employment rate, low education, MW": out["emp_low_educ"]["log_real_mw"],
      "Employment rate, low education, Kaitz": out["emp_low_educ"]["kaitz"]})
R.put("EmpTen", out["emp_low_educ"]["log_real_mw"]["b"], 2)
R.put("EmpSe", out["emp_low_educ"]["log_real_mw"]["se"], 2)
R.put("EmpK", out["emp_low_educ"]["kaitz"]["b"], 2)
R.put("EmpKPboot", out["emp_low_educ"]["kaitz"]["p_boot"], 2)
# Interpretation:
# We find no sign that higher minimum wages reduced employment of the
# low-educated, so selection out of work is unlikely to hide a poverty effect.

# %% 6(g) Material deprivation and what the Kaitz index measures
# Deprivation in the bottom income quintile is dated by survey year. The Kaitz
# index is MW / mean wage, so it rises when the minimum wage rises or when the
# mean wage falls. We split it into log real MW and log real mean wage.
out["depriv_q1"] = run("depriv_q1")
split = {}
for y in ["depriv_q1", "iwp_sal"]:
    d = sample(df, [y, "log_real_mw", "log_real_mean_wage"] + CONTROLS)
    r = fit(twfe(y, ["log_real_mw", "log_real_mean_wage"]), d)
    split[y] = {"mw": coef(r, "log_real_mw", None, 0.1), "mean_wage": coef(r, "log_real_mean_wage", None, 0.1)}
show({"Deprivation, Kaitz": out["depriv_q1"]["kaitz"], "Deprivation, MW part": split["depriv_q1"]["mw"],
      "Deprivation, mean-wage part": split["depriv_q1"]["mean_wage"],
      "In-work poverty, MW part": split["iwp_sal"]["mw"], "In-work poverty, mean-wage part": split["iwp_sal"]["mean_wage"]})
out["kaitz_split"] = split
R.put("DeprK", out["depriv_q1"]["kaitz"]["b"], 2); R.put("DeprKPboot", out["depriv_q1"]["kaitz"]["p_boot"], 2)
R.put("DeprMeanWage", split["depriv_q1"]["mean_wage"]["b"], 1)
R.put("DeprMeanWageSe", split["depriv_q1"]["mean_wage"]["se"], 1)
R.put("DeprMw", split["depriv_q1"]["mw"]["b"], 1, sign=True)
R.put("DeprMwSe", split["depriv_q1"]["mw"]["se"], 1)
R.store("outcomes", out)
R.finish()
# Interpretation:
# A higher Kaitz index goes with more deprivation, which looks odd until we
# split the index: deprivation falls strongly when the mean wage rises and is
# not significantly related (at 5%) to the minimum wage itself. The Kaitz-deprivation
# link is a denominator effect (bad years have low average pay and high
# deprivation), so a change in the Kaitz index is not always a change in
# minimum wage policy.
