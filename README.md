# Replication

| Script (under `data/`) | Results | Output |
|---|---|---|
| `build_panel.py` | The estimation panel from the files in `raw/` | `panel/panel21.csv` |
| `silo1_core.py` | The sample, the estimator, and every test (imported by every stage) | none |
| `silo1_estimates.py` | Table 5 of the paper, Section 7.1, Table 8 of the paper | `estimates.json` |
| `silo1_robustness.py` | Table 6 of the paper, Sections 7.2 and 7.3 | `robustness.json` |
| `silo1_validation.py` | Table 9 of the paper | `validation.json` |
| `silo1_describe.py` | Tables 1 and 3 of the paper, Section 5, the introduction's facts, the Korea facts of Section 8 | `describe.json` |
| `silo1_oos2026.py` | Table 7 of the paper, Section 8 | `oos2026.json` |
| `silo1_paper_figs.py` | Figures 2-11 | `paper_fig_*.pdf` |
| `fetch_bis_turnover.py` | BIS turnover shares, Section 4 | `raw/bis_fx_turnover_shares.csv` |
| `outputs/p20_fig_hypotheses.tex` | Figure 1 | `p20_fig_hypotheses.pdf` |

## Data to download

`data/raw/` holds every input except the VIX, which Cboe distributes. Download two files before running:

- Cboe's daily VIX history, `VIX_History.csv`, from https://www.cboe.com/tradable-products/vix/vix-historical-data/ (direct link: https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv). Save it as `data/raw/cboe_vix_history.csv`. The panel uses its 2001-2025 closes.
- FRED's copy of the same index, series VIXCLS, from https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS. Save it as `data/raw/oos2026/fred_VIXCLS.csv_2026-11-08`. The 2026 out-of-sample check uses its values through 31 October 2026.

## Running

Python 3 with numpy, pandas, scipy, statsmodels, matplotlib, seaborn, and openpyxl. From `data/`, run `python3 build_panel.py`, then `python3 run_pipeline.py`. The pipeline writes its results to `data/outputs/` and one log per stage to `data/rerun_logs/`. `python3 run_pipeline.py --smoke` checks that every stage runs with reduced replication counts. Figure 1 is a TikZ file, `data/outputs/p20_fig_hypotheses.tex`, compiled with `pdflatex`.

## Panel columns

`build_panel.py` writes `data/panel/panel21.csv`: 21 economies, 2001-2025, 525 economy-years, 521 with every regressor.

| Column | Definition |
|---|---|
| `delta_E` | 100 x log change of the end-of-December exchange rate, local currency per US dollar (positive = depreciation) |
| `debt_gdp`, `debt_gdp_lag` | general government gross debt, percent of GDP, and its lag |
| `prim_bal_lag` | primary balance, percent of GDP, lagged |
| `delta_debt_pos_lag`, `delta_debt_neg_lag` | lagged change in debt when positive (else 0), and when negative |
| `prim_bal_streak_lag` | consecutive years of primary-balance decline through the previous year |
| `gdp_growth`, `ca_gdp` | real GDP growth; current account, percent of GDP |
| `infl_diff` | December-on-December CPI inflation (fourth quarter where CPI is quarterly) minus US inflation |
| `rate_spread` | end-of-December central bank policy rate minus the Federal Reserve's (BIS; the US rate is the target-range midpoint) |
| `wui` | World Uncertainty Index, mean of the year's quarters |
| `vix` | mean of the year's daily VIX closes |
| `shadow_rate` | mean of the monthly effective federal funds rate; the Wu-Xia shadow rate in 2009-2015, 2020, and 2021 |
| `pi_stress_annual` | annual mean of the smoothed probability of the high-mean regime of a two-regime Markov switching model of monthly VIX, 2001-2025 |
| `gepu` | Global Economic Policy Uncertainty, current-price weights, mean of the year's months |

The last four columns are global and identical across economies.

## Sources

| File in `data/raw/` | Source |
|---|---|
| `imf_er_eop_2026-11-20.xml` | IMF Exchange Rates, end of period, local currency per US dollar |
| `ifs_cpi.csv` | IMF Consumer Price Index, year-on-year inflation, monthly, downloaded 3 February 2026 |
| `imf_cpi_2026-11-20.xml` | the same IMF dataset: national-base and 2010=100 series, monthly and quarterly |
| `eurostat_hicp_ea_2026-11-20.json` | Eurostat HICP (prc_hicp_minr), all items, annual rate of change, euro area (changing composition) |
| `bis_cbpol_monthly.csv` | BIS central bank policy rates (WS_CBPOL), monthly, end of period, for the 21 economies and the US, from the bulk file of 11 February 2026 |
| `fred_IRSTCI01JPM156N_2026-11-20.csv`, `fred_IRSTCI01IDM156N_2026-11-20.csv` | OECD call-money rate, FRED copy, for the Japanese and Indonesian years without a BIS policy rate |
| `fred_FEDFUNDS_2026-11-20.csv` | Federal Reserve effective federal funds rate, monthly average |
| `weo_oct2026_*.json` | IMF World Economic Outlook, October 2026: gross debt, primary balance, real growth, current account |
| `weo_apr2026_*.json` | IMF World Economic Outlook, April 2026: Korea's debt and primary balance before 2012 |
| `wui.xlsx` | World Uncertainty Index (Ahir, Bloom, and Furceri): quarterly indices (sheet T2) and the authors' GDP weights (sheet T9) |
| `wu_xia_shadow_rate.xlsx` | Wu and Xia shadow federal funds rate, month-end |
| `gepu_2026-11-20.xlsx` | Global Economic Policy Uncertainty Index (Davis), monthly |
| `DEX*.csv` | FRED daily exchange rates behind the introduction's currency facts |
| `bis_fx_turnover_shares.csv` | BIS turnover shares behind the sample rule (written by `fetch_bis_turnover.py`) |
| `areaer_non_floating_spells.csv` | economy-years the IMF classifies as non-floating; `areaer_non_floating_spells_SOURCES.md` gives every spell's dates and sources |
| `oos2026/` | the pinned downloads of the 2026 check: ECB reference rates, BIS policy rates, FRED series, IMF inflation projections, and the WEO snapshot |

## Departures from the default sources

| Economy | Series | Rule |
|---|---|---|
| Australia, New Zealand | CPI | quarterly CPI from the November 2026 IMF download; not in the February 2026 file |
| Czechia, Hungary, Israel, Russia | CPI | the November 2026 IMF download; not in the February 2026 file |
| India, Indonesia, Japan | CPI | the February 2026 file ends in November 2025; December 2025 from the same series, November 2026 |
| Thailand | CPI | the IMF's 2010=100 series, equal to the national-base series wherever both exist; the February 2026 file starts in 2011 and is wrong in 2022-2024 |
| Euro Area | CPI | the official euro-area HICP, changing composition |
| Euro Area | WUI | no euro-area index exists: the 15 members that adopted the euro by 2015, weighted by the authors' GDP weights |
| Japan | policy rate | 2001-2005 and 2013-2015 have no BIS policy rate (the Bank of Japan targeted reserves or the monetary base), so they take the December uncollateralized overnight call rate, its operating target in the other years |
| Indonesia | policy rate | 2001-2004 precede the BI Rate (July 2005), so they take the December overnight call rate |
| Korea | debt, primary balance | the October 2026 WEO starts these series in 2012; earlier years come from April 2026, which agrees at the join within 0.05 points of GDP |

## Methods

- `silo1_core.py` holds the one implementation of the sample, the within estimator, the cluster-robust (CR1) covariance by economy, and every test, so each quantity has one definition.
- Every bootstrap is a wild-cluster bootstrap by economy with the null imposed. Each economy's residuals get one weight per replication: a Rademacher sign (+1 or -1) when the 2^G sign patterns of G economies are at least the B = 9999 replications, and Webb's six-point weights otherwise (the IMF groups). A draw equal to the observed statistic counts as at least as large. B + 1 = 10000 makes the 10, 5, and 1 percent levels whole ranks.
- Each test draws from its own random stream, seeded by the test's name, so adding or removing a test changes no other number. Bootstraps hold the design fixed and resample only the outcome, so replications are solved as matrices.
- H2 and H4 enter debt as a continuous kink, theta_0 D + theta_N (D - gamma) 1{D > gamma}. The grid is 25 quantiles of lagged debt from the 15th to the 85th percentile. H2 is the Wald statistic on theta_N at the least-squares threshold; H4 is the F statistic for separate calm and stressed thresholds against one. Both bootstraps repeat the threshold search in every draw.
- The size-and-power simulation keeps the observed debt paths, plants a kink at the 60th percentile of lagged debt with shifts of 0, 0.5, 1, 1.5, and 2 times the linear slope, and runs the full test (500 panels per row, 999 inner replications). A second simulation gives the errors a shock common to all economies in a year, with the residuals' year share.
- Driscoll-Kraay standard errors sum the scores across economies within a year and use a Bartlett kernel with floor(4 (T/100)^(2/9)) = 2 lags for T = 25.
- The 2026 check freezes the linear model estimated on 2001-2025, feeds it 2026 inputs through 31 October 2026 (observed policy rates, FEDFUNDS, VIX, and WUI to date; WEO projections for growth, the current account, and inflation), and compares each prediction, pro-rated to the share of the year elapsed, with the change in ECB reference rates from 31 December 2025 to 30 October 2026. Downloads are pinned to 8 November 2026; set `SILO1_VINTAGE` to today's date to refresh them.
