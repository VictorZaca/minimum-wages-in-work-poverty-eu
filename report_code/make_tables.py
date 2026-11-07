"""Write the three LaTeX tables of the report from the saved results.

    Table 1  output/tables/descriptives.tex   (step 3)
    Table 2  output/tables/main.tex           (step 4)
    Table 3  output/tables/checks.tex         (steps 5 and 6)
"""
from style import load, write_table


def stars(p):
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.1 else ""


def cell(e, digits=2):
    """'Effect (SE)' and p_boot cells for one coefficient, or two empty cells."""
    if not e:
        return "", ""
    b = f"{e['b']:+.{digits}f}"
    b = b[1:] if float(b) == 0 else b                    # 0.00 rather than -0.00
    return f"${b}$ ({e['se']:.{digits}f})", f"{e['p_boot']:.2f}"


# --- Table 1: descriptive statistics -----------------------------------------------
s3 = load("step3")
lines = []
for r in s3["table1"]:
    label = r["variable"].replace("%", "\\%")
    lines.append(f"{label} & {r['N']:.0f} & {r['Mean']:.1f} & {r['SD']:.1f} & {r['Within SD']:.1f} "
                 f"& {r['Min']:.1f} & {r['Max']:.1f} \\\\")
write_table("descriptives.tex",
            "\\begin{tabular}{lrrrrrr}\n\\toprule\n & $N$ & Mean & SD & Within SD & Min & Max \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")

# --- Table 2: from pooled OLS to two-way fixed effects --------------------------------
ladder = load("step4")["ladder"]
cols = list(ladder)
body = []
for v, label in [("log_real_mw", "Log real minimum wage"), ("kaitz", "Kaitz index (points)")]:
    row_b, row_se, row_p = [label], [""], [""]
    for k in cols:
        e = ladder[k].get(v)
        row_b.append(f"${e['b']:.2f}{stars(e['p'])}$" if e else "")
        row_se.append(f"$({e['se']:.2f})$" if e else "")
        row_p.append(f"$[{e['p_boot']:.2f}]$" if e else "")
    body += [" & ".join(row_b) + " \\\\", " & ".join(row_se) + " \\\\", " & ".join(row_p) + " \\\\[2pt]"]
has = lambda k, s: "Yes" if s in ladder[k]["formula"] else "No"
body.append("\\midrule")
body.append("Country dummies & " + " & ".join(has(k, "C(geo)") for k in cols) + " \\\\")
body.append("Year dummies & " + " & ".join(has(k, "C(year)") for k in cols) + " \\\\")
body.append("Controls & " + " & ".join(has(k, "unemp") for k in cols) + " \\\\")
body.append("Observations & " + " & ".join(str(ladder[k]["n"]) for k in cols) + " \\\\")
body.append("Within $R^2$ & " + " & ".join("" if ladder[k]["within_r2"] is None else f"{ladder[k]['within_r2']:.3f}"
                                           for k in cols) + " \\\\")
body.append("$F$ test, year dummies ($p$) & " + " & ".join(f"{ladder[k]['F_year_p']:.2f}" if "F_year_p" in ladder[k] else ""
                                                        for k in cols) + " \\\\")
header = (" & \\multicolumn{6}{c}{Employees} & \\multicolumn{2}{c}{All employed} \\\\\n"
          "\\cmidrule(lr){2-7}\\cmidrule(lr){8-9}\n & " + " & ".join(cols) + " \\\\")
write_table("main.tex", "\\begin{tabular}{l" + "c" * len(cols) + "}\n\\toprule\n" + header + "\n\\midrule\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n")

# --- Table 3: robustness checks and other outcomes --------------------------------------
s5, s6 = load("step5"), load("step6")["outcomes"]
checks = s5["checks"]
rows_a = []
for key, label in [("no_controls", "No controls"),
                   ("segments", "Separate level after each series break"),
                   ("survey_year", "Minimum wage of the survey year"),
                   ("no_covid", "Without income years 2020--21"),
                   ("no_romania", "Without Romania"),
                   ("q1_deflator", "Deflated by low-income (Q1) prices"),
                   ("kaitz_median", "Kaitz index on the median wage"),
                   ("first_differences", "First differences within segments"),
                   ("wls_employees", "WLS, weights = number of employees"),
                   ("fgls", "FGLS, weights = 1 / country residual variance")]:
    a, pa = cell(checks[key].get("mw"))
    b, pb = cell(checks[key].get("k"))
    n = (checks[key].get("mw") or checks[key].get("k"))["n"]
    rows_a.append(f"{label} & {a} & {pa} & {b} & {pb} & {n} \\\\")
lead = s5["distributed_lag"]
a, pa = cell(lead["log_real_mw"]["lead"])
b, pb = cell(lead["kaitz"]["lead"])
rows_a.append(f"Placebo: next year's minimum wage (lead) & {a} & {pa} & {b} & {pb} & {lead['log_real_mw']['lead']['n']} \\\\")

rows_b = []
for key, label in [("log_real_threshold", "Log real poverty line$^{a}$"),
                   ("pov_anchored", "Poverty, line anchored at its 2008 value$^{b}$"),
                   ("pov_retired", "Poverty of retired people (placebo)$^{c}$"),
                   ("emp_low_educ", "Employment rate, low education$^{d}$"),
                   ("depriv_q1", "Material deprivation, bottom quintile$^{b}$")]:
    e = s6[key]
    a, pa = cell(e["log_real_mw"])
    b, pb = cell(e["kaitz"])
    mean = "" if key == "log_real_threshold" else f" ({e['log_real_mw']['mean']:.0f})"
    rows_b.append(f"{label}{mean} & {a} & {pa} & {b} & {pb} & {e['log_real_mw']['n']} \\\\")

write_table("checks.tex",
            "\\begin{tabular}{lccccc}\n\\toprule\n"
            " & \\multicolumn{2}{c}{Real minimum wage $+10\\%$} & \\multicolumn{2}{c}{Kaitz index $+1$ point} & \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            " & Effect (SE) & $p_{\\text{boot}}$ & Effect (SE) & $p_{\\text{boot}}$ & $N$ \\\\\n\\midrule\n"
            "\\multicolumn{6}{l}{\\textit{A. Robustness. Outcome: in-work poverty of employees (pp)}} \\\\\n"
            + "\n".join(rows_a) +
            "\n\\midrule\n\\multicolumn{6}{l}{\\textit{B. Other outcomes (sample mean in parentheses)}} \\\\\n"
            + "\n".join(rows_b) + "\n\\bottomrule\n\\end{tabular}\n")
print("Tables written to output/tables/")
