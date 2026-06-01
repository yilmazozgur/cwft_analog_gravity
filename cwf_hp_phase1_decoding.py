"""
A3a Phase 1 -- Hayden-Preskill decoding test.

Setup (the cleanest form, capturing the essential physics):
  - k Bell pairs maximally entangle Alice's k qubits with k reference qubits R
  - Alice's k qubits get added to a black hole region of N_BH qubits (initially |0>)
  - Total scrambler region: N = N_BH + k qubits, including Alice's k qubits
  - R sits outside the scrambler region (k extra qubits)
  - Apply the scrambler to the N-qubit region until fully scrambled
  - Reveal qubits one by one (in random order); track I(R, revealed_so_far)

HP claim: I(R, L) jumps from 0 to 2k once |L| crosses a threshold that, for a
random unitary, is just past N/2 (the Page time of the radiation register).
For an OLD black hole (already half-evaporated), the threshold drops to k +
O(log N). The cleaner version below tests the conservation-of-information
property -- the central claim that the substrate's horizon is encoding-based
(info delocalized) rather than destruction-based.

For each (substrate, N_BH, k):
  - Run multiple seeds; average the I(R, L) curve
  - Compare to (i) the Page-curve expectation 2k for |L| >= N - k - O(1)
    and (ii) the HP fast-recovery prediction

This is the load-bearing positive test for Conjecture 1.6.3 (the chapter's
"the remaining route is holographic-QEC"): substrates that conserve info
through unitary scrambling demonstrably support recoverable encoding.
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
    make_sim, stabilizer_matrix, entropy_region, mutual_information,
    prepare_bell_pairs)


# =========================================================================
def run_hp_curve(scrambler_factory, N_BH, k, n_seeds=8, scramble_steps=None,
                 reveal_order_seed=None):
    """Run the HP recovery curve for one (substrate, N_BH, k).

    Returns: (n_revealed_arr, I_mean, I_std, n_total_qubits)
    where n_revealed_arr = [0, 1, ..., N_BH+k] and I_mean[i] is the mean
    mutual info between R and the first i revealed scrambler qubits.
    """
    N = N_BH + k  # scrambler size
    M = N + k  # total qubits (BH + Alice + Reference register R)
    if scramble_steps is None:
        scramble_steps = max(3 * N, 30)  # over-scramble

    I_runs = []
    for s in range(n_seeds):
        sim = make_sim(M, seed=s)
        # Alice's k qubits are 0..k-1 (inside scrambler region)
        # Reference R: qubits N..N+k-1 (outside scrambler region)
        bell_pairs = [(i, N + i) for i in range(k)]
        prepare_bell_pairs(sim, bell_pairs)
        # Scramble the BH region only (qubits 0..N-1)
        scr = scrambler_factory(N, seed=1000 + s)
        for _ in range(scramble_steps):
            scr.step(sim)
        # Reveal qubits in random order
        rng = np.random.default_rng(reveal_order_seed if reveal_order_seed is not None else 7000 + s)
        order = rng.permutation(N).tolist()
        X, Z = stabilizer_matrix(sim, M)
        R = list(range(N, N + k))
        I_curve = np.zeros(N + 1)
        revealed = []
        I_curve[0] = mutual_information(X, Z, R, revealed, M) if revealed else 0.0
        for i in range(N):
            revealed.append(int(order[i]))
            I_curve[i + 1] = mutual_information(X, Z, R, revealed, M)
        I_runs.append(I_curve)
    I_runs = np.array(I_runs)
    return np.arange(N + 1), I_runs.mean(axis=0), I_runs.std(axis=0), M


# =========================================================================
def main():
    print("Phase 1: Hayden-Preskill decoding\n")

    # Each substrate gets its native-feasible set of BH sizes. We pick
    # values such that the total scrambler region N = N_BH + k satisfies
    # the substrate's constraint.
    #   AllToAll, MERA -> N must be power of 2 (MERA) or any (AllToAll)
    #   BrickWall2D, PowerLaw -> N must be perfect square
    ks = [1, 2, 4]
    n_seeds = lambda N: 8 if N <= 100 else 4  # fewer seeds at large N

    # (substrate_name, factory, list of N_BH values)
    # Constraint: N_BH + k must be valid for each k in ks.
    substrates = [
        ("all-to-all",          lambda N, seed: AllToAllScrambler(N, seed=seed),
            [16, 32, 64, 128]),
        # For 2D: N = N_BH + k must be a perfect square for all k in ks.
        # Pick N_BH so that N_BH + 1, +2, +4 hit perfect squares -- usually
        # we instead pick N_BH values such that (N_BH + k) is square for SOME k.
        # Practical: report different N_BH per k.
        # Simpler: include N_BH = square - k for each k separately.
        # Even simpler: pick N total = 16, 36, 64, 100, 144; for each N, all k work.
        # We use N_BH = N - k for each pair.
        ("2D brick-wall",       lambda N, seed: BrickWall2DScrambler(N, seed=seed),
            "square"),  # special: handled below
        ("power-law alpha=2",   lambda N, seed: PowerLawScrambler(N, alpha=2.0, seed=seed),
            "square"),
        ("power-law alpha=1",   lambda N, seed: PowerLawScrambler(N, alpha=1.0, seed=seed),
            "square"),
        ("MERA-tree",           lambda N, seed: MERATreeScrambler(N, seed=seed),
            "pow2"),
    ]
    Ns_square = [16, 36, 64, 100, 144]
    Ns_pow2 = [16, 32, 64, 128]

    all_results = {}
    t_overall = time.time()
    for sub_name, factory, N_BH_set in substrates:
        all_results[sub_name] = {}
        for k in ks:
            # Decide which N values to use for this (substrate, k)
            if N_BH_set == "square":
                N_BH_vals = [N - k for N in Ns_square if N - k > 0]
            elif N_BH_set == "pow2":
                N_BH_vals = [N - k for N in Ns_pow2 if N - k > 0]
            else:
                N_BH_vals = N_BH_set
            for N_BH in N_BH_vals:
                N = N_BH + k
                seeds_here = n_seeds(N)
                # Verify constructible
                try:
                    _ = factory(N, seed=0)
                except (ValueError, AttributeError):
                    continue
                t0 = time.time()
                xs, Im, Is, Mtot = run_hp_curve(factory, N_BH, k, n_seeds=seeds_here)
                dt = time.time() - t0
                target = 2 * k - 1.0
                cross = np.where(Im >= target)[0]
                t_recover = int(cross[0]) if cross.size > 0 else -1
                all_results[sub_name][f"N_BH={N_BH}_k={k}"] = dict(
                    N_BH=N_BH, k=k, N=N, M=Mtot,
                    n_revealed=xs.tolist(),
                    I_mean=Im.tolist(), I_std=Is.tolist(),
                    t_recover=t_recover, target=target,
                    n_seeds=seeds_here, elapsed_s=dt,
                )
                print(f"  [{sub_name:20s}]  N_BH={N_BH:3d} k={k}  "
                      f"N={N:3d}  recovery@{t_recover}/{N}  ({dt:.1f}s, {seeds_here} seeds)")
    print(f"\nPhase 1 total runtime: {time.time() - t_overall:.0f}s")

    # save
    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["A3a_phase1_hp_decoding"] = dict(
        ks=ks,
        Ns_square=Ns_square, Ns_pow2=Ns_pow2,
        by_substrate=all_results,
    )
    json.dump(r_all, open(out, "w"), indent=2)

    # plot: one row per substrate, columns are k values
    fig, axes = plt.subplots(len(substrates), len(ks),
                              figsize=(5.0 * len(ks), 3.2 * len(substrates)),
                              sharex=False, sharey=False)
    if len(substrates) == 1:
        axes = axes[np.newaxis, :]
    cmap = plt.cm.viridis
    for r, (sub_name, _, _) in enumerate(substrates):
        results = all_results.get(sub_name, {})
        for c, k in enumerate(ks):
            ax = axes[r, c]
            recs = sorted([(rec["N_BH"], rec) for key, rec in results.items()
                            if rec["k"] == k])
            colors = cmap(np.linspace(0.1, 0.85, max(len(recs), 1)))
            for (NB, rec), col in zip(recs, colors):
                xs = np.array(rec["n_revealed"])
                Im = np.array(rec["I_mean"])
                Is = np.array(rec["I_std"])
                xs_norm = xs / rec["N"]
                ax.plot(xs_norm, Im / (2 * k), "-", color=col, lw=1.3,
                        label=f"$N_{{BH}}={NB}$")
                ax.fill_between(xs_norm, (Im - Is) / (2 * k), (Im + Is) / (2 * k),
                                color=col, alpha=0.12)
            ax.axhline(1.0, color="k", ls=":", lw=0.6, alpha=0.6)
            ax.axvline(0.5, color="r", ls="--", lw=0.6, alpha=0.4)
            if r == 0:
                ax.set_title(f"k={k}")
            if c == 0:
                ax.set_ylabel(f"{sub_name}\n$I(R,L)/2k$", fontsize=9)
            if r == len(substrates) - 1:
                ax.set_xlabel("$|L|/N$  (fraction revealed)")
            ax.set_ylim(-0.1, 1.15)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=6, loc="upper left", ncol=2)
    fig.suptitle("Phase 1: Hayden-Preskill recovery -- normalized $I(R, L)/2k$ vs revealed fraction",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = os.path.join(os.path.dirname(__file__) or ".",
                             "fig_A3a_phase1_decoding.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {fig_path}")
    print(f"Wrote results.json key: A3a_phase1_hp_decoding")


if __name__ == "__main__":
    main()
