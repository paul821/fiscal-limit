import os

import silo1_core as core

REPS = int(os.environ.get("SILO1_REPS", 500))     # panels per row
B_INNER, MULTIPLES = 999, (0.0, 0.5, 1.0, 1.5, 2.0)

d = core.load()
out = {}

theta0 = core.read("estimates.json")["H1"]["coef"]
sim = core.simulate_threshold_test(d, [k * theta0 for k in MULTIPLES], REPS, B_INNER, core.stream("simulation"))
for row, k in zip(sim["rows"], MULTIPLES):
    row["multiple_of_linear_slope"] = k
    print(f"shift {k:.1f}x: bootstrap rejects {row['reject_10']:.3f} (10%) {row['reject_05']:.3f} (5%); "
          f"textbook {row['textbook_reject_10']:.3f}", flush=True)
out["simulation"] = sim
# same simulation with common year shocks
share = core.read("robustness.json")["residual_comovement"]["year_share"]
simc = core.simulate_threshold_test(d, [k * theta0 for k in MULTIPLES], REPS, B_INNER, core.stream("simulation common shocks"),
                                    year_share=share)
for row, k in zip(simc["rows"], MULTIPLES):
    row["multiple_of_linear_slope"] = k
    print(f"common shocks, shift {k:.1f}x: bootstrap rejects {row['reject_10']:.3f} (10%) {row['reject_05']:.3f} (5%); "
          f"textbook {row['textbook_reject_10']:.3f}", flush=True)
out["simulation_common"] = simc
core.save("validation.json", out)
