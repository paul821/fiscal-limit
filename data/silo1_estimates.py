import silo1_core as core

C = core.CONTROLS
d = core.load()
out = {"n": len(d), "economies": int(d.country.nunique()), "years": [int(d.year.min()), int(d.year.max())], "B": core.B}

# H1 and Appendix A
lin = core.fit(d, "delta_E", ["D"] + C)
h1 = core.wild_test(d, "delta_E", ["D"] + C, "D", rng=core.stream("H1"))
out["H1"] = dict(h1, within_r2=lin["r2"])
rows = [dict(term="D", coef=h1["coef"], se=h1["se"], p_cluster=h1["p_cluster"], p_wild=h1["p_wild"])]
for term in C:
    r = core.wild_test(d, "delta_E", ["D"] + C, term, rng=core.stream(f"appendix A {term}"))
    rows.append(dict(term=term, coef=r["coef"], se=r["se"], p_cluster=r["p_cluster"], p_wild=r["p_wild"]))
out["appendix_A"] = rows
print(f"H1 slope {h1['coef']:.3f}  wild p one-sided {h1['p_wild_upper']:.4f} two-sided {h1['p_wild']:.4f}", flush=True)

# H2
out["H2"] = core.threshold_test(d, C, rng=core.stream("H2"))
print(f"H2 break {out['H2']['gamma']:.4f}  p {out['H2']['p']:.4f}", flush=True)

# H3
f = d.assign(DHS=d.D * d.HS, Dpi=d.D * d.pi)
disc = core.wild_test(f, "delta_E", ["D", "DHS", "HS"] + C, "DHS", rng=core.stream("H3 split"))
cont = core.wild_test(f, "delta_E", ["D", "Dpi", "pi"] + C, "Dpi", rng=core.stream("H3 continuous"))
m = core.fit(f, "delta_E", ["D", "DHS", "HS"] + C)
out["H3"] = dict(split=disc, continuous=cont, slope_calm=float(m["b"]["D"]),
                 slope_stressed=float(m["b"]["D"] + m["b"]["DHS"]), n_stressed=int(d.HS.sum()))
print(f"H3 shift {disc['coef']:.3f}  p one-sided {disc['p_wild_upper']:.4f}", flush=True)

# H4
out["H4"] = core.moving_limit_test(d, C, rng=core.stream("H4"))
print(f"H4 breaks {out['H4']['gamma_calm']:.4f}/{out['H4']['gamma_stress']:.4f}  F {out['H4']['F']:.3f}  p {out['H4']['p']:.4f}", flush=True)

core.save("estimates.json", out)
