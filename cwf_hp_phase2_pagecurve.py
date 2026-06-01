"""
A3a Phase 2 -- Page curves via sequential emission.

For each substrate:
  - Initialize N qubits in |0>; apply scrambler to depth >> t*
  - Sweep subsystem size L from 0 to N; compute S(first L qubits in random order)
  - Page curve: S = min(L, N-L) for fully scrambled Haar-random states
  - Substrate-specific corrections (Clifford has small sub-maximal correction)

The textbook Page curve (Test 3 in the existing framework record) was already
verified for AllToAll-style scrambling on N=40. Phase 2 expands to all four
substrates at multiple N, and compares against Page's theoretical bound to
diagnose whether each substrate reaches a fully-scrambled / random-Clifford-like
state.

This is a sanity-check for the chapter: every substrate that we claim is a
"good scrambler" should produce the textbook Page curve. Deviations from the
Page curve indicate sub-maximal entanglement -- a candidate area-law-like
signal for HaPPY/MERA structures.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stim

sys.path.insert(0, os.path.dirname(__file__) or ".")
from cwf_hp_lib import (
    AllToAllScrambler, BrickWall2DScrambler, PowerLawScrambler, MERATreeScrambler,
    make_sim, stabilizer_matrix, entropy_region)


def page_curve_one_seed(scrambler_factory, N, depth, seed):
    """Apply scrambler for `depth` steps; return S(L) for L=0..N."""
    sim = make_sim(N, seed=seed)
    scr = scrambler_factory(N, seed=10 + seed)
    for _ in range(depth):
        scr.step(sim)
    X, Z = stabilizer_matrix(sim, N)
    rng = np.random.default_rng(1000 + seed)
    order = rng.permutation(N).tolist()
    S = np.zeros(N + 1)
    for L in range(N + 1):
        S[L] = entropy_region(X, Z, order[:L], N)
    return S


def page_curve_avg(scrambler_factory, N, depth, n_seeds):
    S_runs = np.array([page_curve_one_seed(scrambler_factory, N, depth, s)
                       for s in range(n_seeds)])
    return S_runs.mean(axis=0), S_runs.std(axis=0)


def main():
    print("Phase 2: Page curves via sequential emission\n")

    n_seeds = 8
    # For each substrate, native-feasible N values, and a depth deep
    # enough to fully scramble.
    Ns_square = [16, 36, 64, 100, 144]
    Ns_pow2 = [16, 32, 64, 128]
    Ns_general = [16, 32, 64, 128]
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
        all_results[sub_name] = {}
        for N in Ns:
            # Deep over-scramble: ~3 * t*_pessimistic
            depth = max(3 * int(np.ceil(np.sqrt(N))), 30)
            t0 = time.time()
            S_mean, S_std = page_curve_avg(factory, N, depth, n_seeds)
            dt = time.time() - t0
            # Reference: Page-bound min(L, N-L)
            Ls = np.arange(N + 1)
            page_bound = np.minimum(Ls, N - Ls)
            # Deviation from Page (low at all L means we hit the bound)
            deviation = (page_bound - S_mean).clip(min=0)
            mean_dev = float(deviation.mean())
            S_half = float(S_mean[N // 2])
            page_half = N // 2
            all_results[sub_name][f"N={N}"] = dict(
                N=N, depth=depth, n_seeds=n_seeds,
                S_mean=S_mean.tolist(), S_std=S_std.tolist(),
                S_at_half=S_half, page_at_half=page_half,
                mean_deviation_from_page=mean_dev,
                elapsed_s=dt,
            )
            print(f"  [{sub_name:20s}] N={N:3d} depth={depth:3d}  "
                  f"S(N/2)={S_half:5.2f}/{page_half:3d}  "
                  f"mean_dev={mean_dev:.2f}  ({dt:.1f}s)")
    print(f"\nPhase 2 total runtime: {time.time() - t_overall:.0f}s")

    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["A3a_phase2_pagecurves"] = dict(n_seeds=n_seeds, by_substrate=all_results)
    json.dump(r_all, open(out, "w"), indent=2)

    fig, axes = plt.subplots(1, len(substrate_configs),
                              figsize=(4.0 * len(substrate_configs), 4.5),
                              sharey=False)
    if len(substrate_configs) == 1:
        axes = [axes]
    cmap = plt.cm.viridis
    for ax, (sub_name, _, Ns) in zip(axes, substrate_configs):
        colors = cmap(np.linspace(0.1, 0.85, len(Ns)))
        for N, col in zip(Ns, colors):
            rec = all_results[sub_name].get(f"N={N}")
            if rec is None: continue
            Ls = np.arange(N + 1)
            S = np.array(rec["S_mean"])
            page = np.minimum(Ls, N - Ls)
            ax.plot(Ls / N, S / (N / 2), "-", color=col, lw=1.5, label=f"N={N}")
            ax.plot(Ls / N, page / (N / 2), ":", color=col, lw=0.7, alpha=0.7)
        ax.axhline(1.0, color="k", ls=":", lw=0.5)
        ax.axvline(0.5, color="r", ls="--", lw=0.6, alpha=0.4)
        ax.set_title(sub_name)
        ax.set_xlabel("L/N")
        ax.set_ylabel("S/(N/2)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Phase 2: normalized Page curves -- all 4 substrates", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(os.path.dirname(__file__) or ".",
                             "fig_A3a_phase2_pagecurves.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {fig_path}")
    print(f"Wrote results.json key: A3a_phase2_pagecurves")


if __name__ == "__main__":
    main()
