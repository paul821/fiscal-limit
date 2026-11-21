import io, os, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://stats.bis.org/api/v1/data/WS_DER_OTC_TOV/A.U.A.B.5J.A.5J.A..TO1.A.A.3.C?startPeriod=2013&format=csv"
raw = subprocess.run(["curl", "-sL", "--retry", "3", "--max-time", "180", URL], capture_output=True, text=True, check=True).stdout
d = pd.read_csv(io.StringIO(raw)); years = [2013, 2016, 2019, 2022, 2025]; d = d[d.TIME_PERIOD.isin(years)]
p = d.pivot(index="DER_CURR_LEG1", columns="TIME_PERIOD", values="OBS_VALUE")
s = p / p.loc["TO1"] * 100; s["avg"] = s[years].mean(axis=1); s["avg_of_total"] = s["avg"] / 2
s.round(4).to_csv(os.path.join(HERE, "raw", "bis_fx_turnover_shares.csv"))
print(s.loc[["NZD", "HUF", "CZK", "ILS", "IDR", "THB", "CLP"], "avg"].round(2).to_dict())
