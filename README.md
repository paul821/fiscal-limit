# Replication

Every result comes from one pipeline (Table 11), and all listed paths are relative to the `data` folder. `core.py` contains the implementation of the sample, the estimator, and every hypothesis test; every pipeline stage imports it. `run_pipeline.py` runs the stages in order. Each bootstrap test has its own fixed random seed, so a rerun reproduces the same draws. The 2026 out-of-sample downloads are pinned to the retrieval date of 8 November 2026, and the World Economic Outlook snapshot dates from 16 October 2026.

The estimation panel comes from `build_panel.py`, which reads the files in `raw/`. Each series takes, year by year, the first source in the economy's list that has a value. The default sources include the IMF's end-of-December exchange rate and December consumer price inflation (International Monetary Fund, 2026b), the end-of-December central bank policy rate (Bank for International Settlements, 2026), and the October 2026 World Economic Outlook (International Monetary Fund, 2026a).

Most fallbacks cover an economy or a year that the default source lacks. For instance, Japan and Indonesia lack a policy rate in some years, so those years take the overnight call rate (Federal Reserve Bank of St. Louis, 2026). The Euro Area's inflation is the official euro-area index (Eurostat, 2026). We also note that the Euro Area's World Economic Outlook series are the euro-area aggregate. The October 2026 database omits Korea's gross debt and primary balance before 2012, so those years come from the April 2026 version.

The script also estimates the Markov-switching model for $\hat{\pi}_t$, and it replaces the federal funds rate with the Wu-Xia shadow rate in 2009-2015, 2020, and 2021, as these are the years in which the zero lower bound held for most of the year.

Table 11: Replication map. Outputs are written to the file in the last column (under `outputs/` unless specified otherwise). `build_panel.py`, `fetch_bis_turnover.py`, and the Figure 1 source are run by hand before it.

| Script (under `data/`) | Results | Output |
|---|---|---|
| `build_panel.py` | The estimation panel from the files in `raw/` | `panel/panel21.csv` |
| `core.py` | The sample, the estimator, and every test (imported by every stage) | none |
| `estimates.py` | Table 5, Section 7.1, Table 8 | `estimates.json` |
| `robustness.py` | Table 6, Sections 7.2 and 7.3 | `robustness.json` |
| `validation.py` | Table 9 | `validation.json` |
| `describe.py` | Tables 1 and 3, Section 5, the introduction's facts, the Korea facts of Section 8 | `describe.json` |
| `oos2026.py` | Table 7, Section 8 | `oos2026.json` |
| `paper_figs.py` | Figures 2-11 | `paper_fig_*.pdf` |
| `paper_numbers.py` | The bootstrap and simulation numbers in the text | `paper_numbers.json` |
| `fetch_bis_turnover.py` | BIS turnover shares, Section 4 | `raw/bis_fx_turnover_shares.csv` |
| `outputs/fig_hypotheses.tex` | Figure 1 | `fig_hypotheses.pdf` |
