# Replication

Every result comes from one pipeline (Table 11 of the paper), and all listed paths are relative to the `data` folder. `silo1_core.py` contains the implementation of the sample, the estimator, and every hypothesis test; every pipeline stage imports it. `run_pipeline.py` runs the stages in order. Every bootstrap test draws from its own random stream. The 2026 out-of-sample downloads are pinned to the retrieval date of 8 November 2026, and the World Economic Outlook snapshot dates from 16 October 2026.

The estimation panel comes from `build_panel.py`, which reads the files in `raw/`. Each series takes, year by year, the first source in the economy's list that has a value. The default sources include the IMF's end-of-December exchange rate and December consumer price inflation (IMF International Financial Statistics), the end-of-December central bank policy rate (BIS central bank policy rates), and the October 2026 World Economic Outlook (IMF World Economic Outlook, October 2026). 

Most fallbacks cover an economy or a year that the default source lacks. For instance, Japan and Indonesia lack a policy rate in some years, so those years take the overnight call rate (FRED). The Euro Area's inflation is the official euro-area index (Eurostat HICP). We also note that the Euro Area's World Economic Outlook series are the euro-area aggregate. The October 2026 database omits Korea's gross debt and primary balance before 2012, so those years come from the April 2026 version. 

The script also estimates the Markov-switching model for the stress probability, and it replaces the federal funds rate with the Wu-Xia shadow rate in 2009-2015, 2020, and 2021, as these are the years in which the zero lower bound held for most of the year.

Replication map. Outputs are written to the file in the last column (under `outputs/` unless specified otherwise). `build_panel.py`, `fetch_bis_turnover.py`, and the Figure 1 source are run by hand before it.

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
