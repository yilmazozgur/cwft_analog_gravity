"""
A3a Phase 4 -- scrambling-time scaling across substrates.

For each of the 4 substrate architectures, measure t*(N): the number of
scrambler steps required for S(left half) to reach >= 95% of the Page bound
N/2.

Expected scaling:
  AllToAll          : t* ~ log N  (canonical fast scrambler)
  2D brick-wall     : t* ~ sqrt(N) = L  (local 2D ballistic)
  Power-law         : between log N and sqrt(N), depending on alpha
  MERA-tree         : t* ~ const O(1) steps  (one tree pass scrambles)

Each "step" applies ~N/2 random Clifford 2-qubit gates, so cross-substrate
comparison in gate-count is approximately uniform per step.

This is the framework's regime-baseline for the HP test: substrates that fail
to scramble in poly(N) steps are not viable as gravity-ready substrates.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__) or ".")
from cwf_hp_lib import (
    AllToAllScrambler, BrickWall2DScrambler, PowerLawScrambler, MERATreeScrambler,
    scrambling_time)


def main():
    print("Phase 4: scrambling-time scaling t*(N)\n")

    Ns_square = [16, 36, 64, 100, 144]
    Ns_pow2 = [16, 32, 64, 128]
    Ns_general = [16, 32, 64, 128]
    n_seeds = 8

    substrate_configs = [
        ("all-to-all",        lambda N, seed: AllToAllScrambler(N, seed=seed),     Ns_general),
        ("2D brick-wall",     lambda N, seed: BrickWall2DScrambler(N, seed=seed),  Ns_square),
        ("power-law alpha=2", lambda N, seed: PowerLawScrambler(N, alpha=2.0, seed=seed), Ns_square),
        ("power-law alpha=1", lambda N, seed: PowerLawScrambler(N, alpha=1.0, seed=seed), Ns_square),
        ("MERA-tree",         lambda N, seed: MERATreeScrambler(N, seed=seed),     Ns_pow2),
    ]

    all_results = {}
    t_overall = time.time()
    for sub_name, factory, Ns in substrate_configs:
        all_results[sub_name] = []
        for N in Ns:
            t0 = time.time()
            ts = [scrambling_time(factory(N, seed=s), max_steps=4 * N + 100,
                                    seed=10 + s) for s in range(n_seeds)]
            tmed = float(np.median(ts))
            tstd = float(np.std(ts))
            dt = time.time() - t0
            all_results[sub_name].append(dict(
                N=N, t_star_med=tmed, t_star_std=tstd, t_star_all=ts,
                elapsed_s=dt
            ))
            print(f"  [{sub_name:20s}] N={N:3d}  t*={tmed:5.1f}+-{tstd:4.1f}  ({dt:.1f}s)")

    print(f"\nPhase 4 total runtime: {time.time() - t_overall:.0f}s")

    # Fit scaling
    print("\n--- Scaling fits ---")
    for sub_name, recs in all_results.items():
        Ns = np.array([r["N"] for r in recs])
        ts = np.array([r["t_star_med"] for r in recs])
        if len(Ns) >= 3:
            # try log: t* = a + b log N
            log_p = np.polyfit(np.log(Ns), ts, 1)
            log_res = np.sum((ts - np.polyval(log_p, np.log(Ns))) ** 2)
            # try sqrt: t* = a + b sqrt(N)
            sqrt_p = np.polyfit(np.sqrt(Ns), ts, 1)
            sqrt_res = np.sum((ts - np.polyval(sqrt_p, np.sqrt(Ns))) ** 2)
            # try const
            const_res = np.sum((ts - ts.mean()) ** 2)
            best = "log" if log_res < sqrt_res and log_res < const_res else (
                   "sqrt" if sqrt_res < const_res else "const")
            print(f"  [{sub_name:20s}]  log_res={log_res:.1f}  "
                  f"sqrt_res={sqrt_res:.1f}  const_res={const_res:.1f}  --> {best}")

    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["A3a_phase4_scrambling_time"] = dict(n_seeds=n_seeds, by_substrate=all_results)
    json.dump(r_all, open(out, "w"), indent=2)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(len(substrate_configs))]
    for (sub_name, _, _), col in zip(substrate_configs, colors):
        recs = all_results.get(sub_name, [])
        if not recs: continue
        Ns = np.array([r["N"] for r in recs])
        ts = np.array([r["t_star_med"] for r in recs])
        tstd = np.array([r["t_star_std"] for r in recs])
        axes[0].errorbar(Ns, ts, yerr=tstd, fmt="o-", color=col,
                          lw=1.5, ms=6, capsize=3, label=sub_name)
        axes[1].errorbar(Ns, ts, yerr=tstd, fmt="o-", color=col,
                          lw=1.5, ms=6, capsize=3, label=sub_name)
    axes[0].set_xlabel("N")
    axes[0].set_ylabel(r"$t_*$ (steps)")
    axes[0].set_title("Linear scale")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("N (log)")
    axes[1].set_ylabel(r"$t_*$ (log)")
    axes[1].set_title("Log-log (slope 1 = ballistic; slope 1/2 = 2D local; slope 0 = fast)")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)
    fig.suptitle("Phase 4: scrambling-time scaling across substrates", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(os.path.dirname(__file__) or ".",
                             "fig_A3a_phase4_scrambling.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
