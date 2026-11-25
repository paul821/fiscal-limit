from decimal import Decimal, ROUND_HALF_UP

import core as core


def f(x, dp=3):
    return str(Decimal(repr(round(float(x), 10))).quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


pct = lambda x: f(100 * x, 1)
E, R, V, S = (core.read(n) for n in ("estimates.json", "robustness.json", "validation.json", "describe.json"))
s = R["samples"]
n = {
    "H1_P1": f(E["H1"]["p_wild_upper"]), "H1_P2": f(E["H1"]["p_wild"]), "H1_CLP1": f(E["H1"]["p_cluster_upper"]),
    "HDK_P1": f(R["driscoll_kraay"]["p_upper"]),
    "HFLOAT_P1": f(R["floating_years"]["p_wild_upper"]),
    "HJ01_P1": f(s["no_influential"]["H1"]["p_wild_upper"]),
    "HNOJ_P1": f(s["no_japan"]["H1"]["p_wild_upper"]),
    "H2_P": f(E["H2"]["p"]), "H2J01_P": f(s["no_influential"]["H2"]["p"]),
    "H2NJ_P": f(s["no_japan"]["H2"]["p"]),
    "H3D_P1": f(E["H3"]["split"]["p_wild_upper"]), "H3D_P2": f(E["H3"]["split"]["p_wild"]),
    "H3C_P1": f(E["H3"]["continuous"]["p_wild_upper"]),
    "H3G_P1": f(R["text_measures"]["gepu"]["p_wild_upper"]), "H3W_P1": f(R["text_measures"]["wui"]["p_wild_upper"]),
    "H3NJ_P1": f(s["no_japan"]["H3"]["split"]["p_wild_upper"]), "H3NJC_P1": f(s["no_japan"]["H3"]["continuous"]["p_wild_upper"]),
    "H3J01_P1": f(s["no_influential"]["H3"]["split"]["p_wild_upper"]),
    "H4_P": f(E["H4"]["p"]), "H4_CRIT90": f(E["H4"]["F_crit_90"], 2),
    "H4J01_P": f(s["no_influential"]["H4"]["p"]), "H4NJ_P": f(s["no_japan"]["H4"]["p"]),
    "G_IMFADV_P": f(R["groups"]["imf_advanced"]["p"]), "G_IMFEM_P": f(R["groups"]["imf_emerging"]["p"]),
    "KOR_SD": f(next(r for r in S["table_countries"] if r["country"] == "Korea")["sd_dep"], 2),
    "INF1": f(R["influence"][0]["slope_change"], 2), "INF2": f(R["influence"][1]["slope_change"], 2),
    "INF1_WHO": f'{R["influence"][0]["country"]} {R["influence"][0]["year"]}',
    "INF2_WHO": f'{R["influence"][1]["country"]} {R["influence"][1]["year"]}',
    "DSTEP": f(S["within"]["debt_step_stressed"], 1), "YEAR_SHARE": f(100 * R["residual_comovement"]["year_share"], 0),
    "RES_CORR": f(R["residual_comovement"]["mean_pairwise_corr"], 2),
    "CR_MIN": str(S["crossings"]["min"]), "CR_MAX": str(S["crossings"]["max"]), "CR_MED": str(int(S["crossings"]["median"])),
    "EC_MIN": str(S["crossings"]["economies_min"]), "EC_MAX": str(S["crossings"]["economies_max"]),
}
rows = V["simulation"]["rows"]
n["SIZE10"], n["NAIVE10"] = pct(rows[0]["reject_10"]), pct(rows[0]["textbook_reject_10"])
n["power"] = [dict(shift=f(r["shift"], 1), multiple=f(r["multiple_of_linear_slope"], 1).rstrip("0").rstrip("."),
                   reject_10=pct(r["reject_10"]), reject_05=pct(r["reject_05"])) for r in rows]
for r in E["appendix_A"]:
    n["A_P_" + r["term"]] = f(r["p_wild"])
n["SIM_REPS"], n["SIM_B_INNER"], n["B"] = str(V["simulation"]["design"]["reps"]), str(V["simulation"]["design"]["B_inner"]), str(E["B"])
d = core.load()
n["MAXD_NOJ"] = f(d[d.country != "Japan"].debt_gdp_lag.max(), 1)
core.save("paper_numbers.json", n)
print({k: v for k, v in n.items() if k != "power"})
print("power:", n["power"])
