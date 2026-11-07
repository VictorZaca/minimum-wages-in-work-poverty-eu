# Minimum wages and in-work poverty in the EU

Final project for Econometrics (ECO_4MO11), Télécom Paris, November 2026.
Yuewen Sun, Victor Zacarias and Robin Herment.

**Question.** Across EU countries with a statutory minimum wage (2009–2024), is a higher real or relative minimum wage associated with lower in-work poverty among employees, while controlling for permanent country-specific differences and common shocks?

**Short answer.** Not in a way we can detect for the real minimum wage. In a two-way fixed effects panel of 21 countries, a 10% higher real minimum wage goes with a change of −0.03 percentage points in employee in-work poverty (95% CI −0.29 to +0.24). The Kaitz index (minimum wage / mean wage) has a small negative association, −0.05 points per Kaitz point, which becomes significant only with weighted estimators (WLS and FGLS, wild bootstrap p ≈ 0.02–0.03). Both are much smaller than the 12–13% reduction in the simulations behind the EU Directive.

## Where to look

| File | What it is |
|---|---|
| [`report/main.pdf`](report/main.pdf) | The report (5 pages) |
| [`analysis.ipynb`](analysis.ipynb) | The whole analysis as a notebook, with every output. Start here to read the code |
| [`analysis/`](analysis/) | The same analysis as six scripts, written in cells like the course labs |
| [`report_code/`](report_code/) | Turns the saved results into the report's tables and figures, and builds the notebook |
| [`presentation/slides.pdf`](presentation/slides.pdf) | The slides of the final presentation |

## Reproduce everything

Python 3.12 with the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
python run_all.py                 # uses the Eurostat snapshot in data/raw/ (about a minute)
python run_all.py --download      # downloads fresh data from the Eurostat API first
python run_all.py --no-notebook   # skips re-executing analysis.ipynb
```

To rebuild the PDF (any LaTeX distribution; we used [Tectonic](https://tectonic-typesetting.github.io/)):

```bash
cd report
tectonic main.tex                 # or: latexmk -pdf main.tex
```

Every number in the text of the report is written by the analysis scripts to `output/results/step*.tex` as LaTeX macros (`\newcommand`, the same idea as `results.tex` in the course labs) and read by `report/main.tex`. Nothing is typed by hand, so the report always matches the code. The bootstrap uses a fixed seed (`SEED` in `analysis/tools.py`), so reruns give identical numbers. Re-downloading may change them slightly, because Eurostat revises its data.

## Repository structure

```
run_all.py                      runs everything below, in order
requirements.txt
analysis.ipynb                  notebook built from analysis/*.py and executed
analysis/
  tools.py                      paths, sample, model formula, cluster-robust OLS/WLS, wild cluster bootstrap
  01_get_data.py                Eurostat API -> data/raw/ (only with --download; otherwise describes the snapshot)
  02_build_panel.py             data/panel.csv: income-year dating, real minimum wage, low-income price index
  03_descriptives.py            summary statistics, France vs Germany, long differences, Romania
  04_main_regressions.py        pooled OLS to two-way FE, Kaitz, all employed, comparison with the simulations
  05_diagnostics_robustness.py  Breusch-Pagan/White, AR(1), WLS/FGLS, robustness checks, lead test, Cook's distance
  06_mechanisms.py              poverty lines, household types, poverty threshold, placebo, selection, deprivation
report_code/
  make_tables.py                Tables 1-3 (LaTeX) from output/results/
  make_figures.py               Figures 1-3
  make_slide_figures.py         figures for the presentation (output/figures/slides/)
  make_notebook.py              builds and runs analysis.ipynb from the scripts
  style.py                      paths, colours, figure style
data/
  raw/                          Eurostat snapshot (7 November 2026) and _metadata.json
  panel.csv                     21 countries x income years 2009-2024
output/
  results/                      step*.json (all estimates) and step*.tex (numbers quoted in the report)
  tables/, figures/             what the report includes
  figures/slides/               figures used in the presentation
  analysis_plots/               quick diagnostic plots from the analysis scripts
report/
  main.tex, references.bib, main.pdf
presentation/
  slides.pdf                    the slides of the final presentation
```

| In the report | Produced by |
|---|---|
| Numbers in the text | `analysis/02`–`06` → `output/results/step*.tex` |
| Table 1, Figure 1 | `analysis/03_descriptives.py` → `report_code/make_tables.py`, `make_figures.py` |
| Table 2, Figure 2 | `analysis/04_main_regressions.py`, `05_diagnostics_robustness.py` → `report_code/` |
| Table 3 | `analysis/05_diagnostics_robustness.py`, `06_mechanisms.py` → `report_code/make_tables.py` |
| Figure 3 | `analysis/06_mechanisms.py` → `report_code/make_figures.py` |

## Data

All series come from the [Eurostat database](https://ec.europa.eu/eurostat/web/main/data/database) (free reuse with acknowledgement of the source). `data/raw/_metadata.json` records the filters, the download date and Eurostat's last update for each table.

| Variable | Eurostat table | Definition |
|---|---|---|
| In-work poverty, employees (main outcome) | `ilc_iw01` (wstatus=SAL) | Employees aged 18–64 in work for at least half of the income reference year, whose household equivalised disposable income is below 60% of the national median (%) |
| In-work poverty, all employed | `ilc_iw01` (wstatus=EMP) | Same, self-employed included |
| By working time / household type | `ilc_iw07`, `ilc_iw02` | Same, all employed, by full-/part-time and household composition |
| Poverty at 40/50/60/70% of the median | `ilc_li04` | By most frequent activity status: employees, retired, unemployed, aged 18+ (from income year 2014) |
| Minimum wage | `earn_mw_cur` | Gross monthly statutory minimum wage, average of the January and July values |
| Kaitz index | `earn_mw_avgr2` | Minimum wage as % of mean (and median) gross monthly earnings |
| Prices | `prc_hicp_aind` | HICP, annual average index (2015 = 100), all items and the 12 COICOP divisions |
| Budget shares | `hbs_str_t223` | Consumption structure of the bottom and top income quintiles (Household Budget Survey) |
| Controls | `une_rt_a`, `nama_10_pc`, `lfsa_eppga` | Unemployment rate (15–74), real GDP per capita (chain-linked 2020 euro), part-time share (15–64) |
| Weights | `lfsa_eegaed` | Number of employees aged 20–64 (thousands) |
| Employment of the low-educated | `lfsa_ergaed` | Employment rate, aged 20–64, ISCED 0–2 |
| Poverty threshold | `ilc_li01` | 60% of the median equivalised income, single adult, national currency |
| Anchored poverty | `ilc_li22b` | Poverty rate (18–64) against the 2008 line kept constant in real terms |
| Deprivation | `ilc_mdsd02` | Material and social deprivation rate, bottom income quintile |

**Sample.** The 21 EU member states with a national statutory minimum wage: BE, BG, CZ, DE, EE, EL, ES, FR, HR, HU, IE, LT, LU, LV, MT, NL, PL, PT, RO, SI, SK. Germany enters in 2015. 330 country-years in the main regressions.

**Timing.** EU-SILC wave *w* measures income of calendar year *w*−1 (Ireland since 2020), so poverty outcomes are dated by income year and matched with the minimum wage and controls of that year. Material deprivation describes conditions at the interview and keeps its survey-year dating.

**Real minimum wage.** Nominal minimum wage in national currency divided by the country's own HICP, expressed in 2015 purchasing power standards. As a check we also deflate it by a fixed-weight price index built from the budget shares of the poorest income quintile.

## Methods and the course

The main model is a two-way fixed effects regression:

```
IWP_it = beta * MW_it + gamma' X_it + alpha_i + delta_t + u_it
```

MW is the log real minimum wage or the Kaitz index, X are the controls, and alpha and delta are country and year dummies. Standard errors are clustered by country with t(20) critical values. Because there are only 21 clusters, every coefficient of interest also has a wild cluster bootstrap p-value (restricted residuals, Webb six-point weights, 9,999 draws; Cameron, Gelbach and Miller 2008; Webb 2023), implemented in `analysis/tools.py`.

| Lecture | Where it is used |
|---|---|
| 1–2: omitted variables, dummy variables | pooled OLS vs country and year dummies (`04`, cell 4(a)) |
| 3: t and F tests, intervals, power | F tests of year dummies and controls, minimum detectable effect (`04`, 4(b)) |
| 4: asymptotics | few clusters: wild cluster bootstrap vs the usual cluster-robust test (`tools.py`, `05`, 5(c)) |
| 5: functional form, interactions, influence | log outcome, quadratic Kaitz, standardized coefficients, Cook's distance, leave-one-out (`05`, 5(g)–(h)); stacked interaction (`06`, 6(c)) |
| 6: heteroskedasticity, WLS/FGLS | Breusch–Pagan and special White tests, WLS and FGLS (`05`, 5(a), 5(c)) |
| 7: dynamics, strict exogeneity, persistence | residual AR(1), distributed lag with a lead, first differences (`05`, 5(b), 5(e)–(f)) |

## Contributions

Yuewen Sun and Robin Herment prepared the project proposal. Yuewen led the France and Germany comparison and the motivating evidence, Robin led the policy background and the literature, and Victor Zacarias led the data work, the code and the regressions. All three of us discussed the specification, wrote and revised the report, and prepared the slides and the presentation together.

## Use of AI

We used an AI assistant (Claude, Anthropic) to help write and check the code, to search for and check references, and to edit the English of the report. We checked the sources and the results ourselves.
