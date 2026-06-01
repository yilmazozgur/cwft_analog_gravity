"""
A3a Phase 3b -- HaPPY-like tree code: area-law via QEC encoding.

The chapter's Conjecture 1.6.3 says the only substrate class that can satisfy
all Einstein-dynamics conditions (universal area-law entropy, reversibility,
fast scrambling, non-dissipative horizon mechanism) is a holographic-QEC
class -- a substrate whose entanglement is determined by a tensor-network /
QEC code structure rather than by random scrambling.

Phase 3a (the generic-substrates baseline, separate file) confirms that
fully-scrambled generic substrates have volume-law entropy S(L) = Page bound
= min(L, N-L). This is the volume-law / Page behaviour, NOT area-law.

Phase 3b tests whether a *code-structured* substrate -- a Clifford QEC
tree code -- has sub-Page entanglement, the hallmark of area-law.

Construction: a "doubling tree code".
  - Start with k_bulk "bulk" qubits in |0>.
  - At each tree layer, pair each currently-active qubit with one new
    ancilla (initially |0>) and apply a random 2-qubit Clifford. Active
    set doubles each layer.
  - After n_layers, we have N = k_bulk * 2^n_layers boundary qubits.

For a fully-scrambled state, S(contiguous L qubits) = min(L, N-L).
For the tree code state, the entanglement of a boundary region A is bounded
by the number of TREE CUTS that A makes -- O(log |A|) for a contiguous A,
giving S ~ log L (area-law in 1D).

Comparison:
  - Random Clifford state: S(L) = min(L, N-L) -- volume law
  - Tree code state: S(L) ~ log L -- area law (for 1D boundary)

If the tree code gives sub-Page S(L) with universal slope across (k_bulk, N),
that's empirical evidence for Conjecture 1.6.3: an encoding-based substrate
class exists with the area-law property.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stim

sys.path.insert(0, os.path.dirname(__file__) or ".")
from cwf_hp_lib import make_sim, stabilizer_matrix, entropy_region, gf2_rank


# =========================================================================
def build_tree_code_state(k_bulk, n_layers, seed):
    """Build a tree-doubling code state on N = k_bulk * 2^n_layers qubits.

    Layer 0: k_bulk active qubits (bulk), all in |0>.
    Layer t: pair each active qubit with a new ancilla; apply random Clifford.
             Active set doubles.

    Returns: (sim, N, leaf_order) where leaf_order is the boundary qubit
    indexing such that leaf_order[i] is the i-th boundary qubit in a
    natural ordering aligned with the tree structure.
    """
    N = k_bulk * (2 ** n_layers)
    sim = make_sim(N, seed=seed)
    # active: ordered list of "current active qubits". Initially the bulk.
    active = list(range(k_bulk))
    next_q = k_bulk
    for layer in range(n_layers):
        new_active = []
        for q in active:
            a = next_q; next_q += 1
            tab = stim.Tableau.random(2)
            sim.do_tableau(tab, [int(q), int(a)])
            new_active.append(q)
            new_active.append(a)
        active = new_active
    return sim, N, active


def random_clifford_state(N, depth, seed):
    """For comparison: fully-scrambled random Clifford state."""
    sim = make_sim(N, seed=seed)
    rng = np.random.default_rng(seed + 12345)
    for d in range(depth):
        perm = rng.permutation(N)
        for i in range(N // 2):
            a, b = int(perm[2 * i]), int(perm[2 * i + 1])
            tab = stim.Tableau.random(2)
            sim.do_tableau(tab, [a, b])
    return sim


def measure_contiguous_S(sim, N, leaf_order, max_L=None):
    """For a given state, compute S(contiguous region of size L) for
    L = 0, 1, ..., N. Contiguous means qubits in positions
    leaf_order[0], leaf_order[1], ..., leaf_order[L-1].
    """
    X, Z = stabilizer_matrix(sim, N)
    if max_L is None:
        max_L = N
    Ls = np.arange(0, max_L + 1)
    S = np.zeros(len(Ls))
    for k, L in enumerate(Ls):
        region = leaf_order[:L]
        S[k] = entropy_region(X, Z, region, N)
    return Ls, S


# =========================================================================
def main():
    print("Phase 3b: HaPPY-like tree code area-law test\n")

    n_seeds = 12

    # Tree codes: vary k_bulk and n_layers
    # (k_bulk, n_layers, N) - N must be modest for cubic-time entropy compute
    configs = [
        ("k=1, N=16", 1, 4),
        ("k=1, N=32", 1, 5),
        ("k=1, N=64", 1, 6),
        ("k=1, N=128", 1, 7),
        ("k=2, N=32", 2, 4),
        ("k=2, N=64", 2, 5),
        ("k=4, N=64", 4, 4),
        ("k=4, N=128", 4, 5),
    ]
    # also compare random Clifford (full Page curve) at matched N
    Ns_compare = [16, 32, 64, 128]

    all_results = {}
    t_overall = time.time()
    for label, k_bulk, n_layers in configs:
        N = k_bulk * (2 ** n_layers)
        S_runs = []
        t0 = time.time()
        for s in range(n_seeds):
            sim, _, leaves = build_tree_code_state(k_bulk, n_layers, seed=s)
            Ls, S = measure_contiguous_S(sim, N, leaves)
            S_runs.append(S)
        S_arr = np.array(S_runs)
        S_mean = S_arr.mean(axis=0); S_std = S_arr.std(axis=0)
        page = np.minimum(np.arange(N + 1), N - np.arange(N + 1))
        dev = (page - S_mean).clip(min=0)
        max_S = float(S_mean.max())
        all_results[label] = dict(
            k_bulk=k_bulk, n_layers=n_layers, N=N,
            S_mean=S_mean.tolist(), S_std=S_std.tolist(),
            max_S=max_S, deviation_total=float(dev.sum()),
            elapsed_s=time.time() - t0,
        )
        dt = time.time() - t0
        print(f"  [tree {label:13s}] k={k_bulk} N={N:3d}  "
              f"S_max={max_S:5.2f} (page={N//2}; ratio={max_S/(N//2):.2f})  "
              f"deviation_sum={dev.sum():.1f}  ({dt:.1f}s)")

    # Random Clifford (Page-curve) reference at matched N
    rc_results = {}
    for N in Ns_compare:
        depth = max(3 * int(np.ceil(np.sqrt(N))), 30)
        S_runs = []
        for s in range(n_seeds):
            sim = random_clifford_state(N, depth, seed=s)
            X, Z = stabilizer_matrix(sim, N)
            Ls = np.arange(0, N + 1)
            S = np.array([entropy_region(X, Z, list(range(L)), N) for L in Ls])
            S_runs.append(S)
        S_arr = np.array(S_runs)
        rc_results[f"N={N}"] = dict(
            N=N, depth=depth,
            S_mean=S_arr.mean(axis=0).tolist(),
            S_std=S_arr.std(axis=0).tolist(),
        )
        print(f"  [random  N={N:3d}]    S(N/2)={S_arr.mean(axis=0)[N//2]:.2f}  (Page bound N/2={N//2})")

    print(f"\nPhase 3b total runtime: {time.time() - t_overall:.0f}s")

    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["A3a_phase3b_happy_tree"] = dict(
        n_seeds=n_seeds,
        tree_configs=all_results,
        random_clifford_baseline=rc_results,
    )
    json.dump(r_all, open(out, "w"), indent=2)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # Left: S(L) for tree codes (normalized by N/2 -- the Page bound)
    ax = axes[0]
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.05, 0.85, len(configs)))
    for (label, k_b, nl), col in zip(configs, colors):
        rec = all_results[label]
        N = rec["N"]
        S = np.array(rec["S_mean"])
        Sstd = np.array(rec["S_std"])
        Ls = np.arange(N + 1)
        ax.plot(Ls / N, S, "-", color=col, lw=1.4, label=label)
        ax.fill_between(Ls / N, S - Sstd, S + Sstd, color=col, alpha=0.1)
    # overlay random-Clifford Page curve at N=64 for comparison
    rec64 = rc_results.get("N=64")
    if rec64:
        Ls = np.arange(rec64["N"] + 1)
        ax.plot(Ls / rec64["N"], np.array(rec64["S_mean"]), "k--",
                lw=1.0, alpha=0.6, label="random Clifford N=64 (Page)")
    ax.set_xlabel(r"$L/N$ (contiguous region fraction)")
    ax.set_ylabel(r"$S(L)$ [bits]")
    ax.set_title("Tree-code S(L) vs random Clifford")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Right: max S (across L) vs k_bulk -- should saturate at ~k_bulk
    ax = axes[1]
    points_k = sorted(set([rec["k_bulk"] for rec in all_results.values()]))
    for k in points_k:
        recs = sorted([(rec["N"], rec["max_S"]) for rec in all_results.values()
                       if rec["k_bulk"] == k])
        Ns = np.array([r[0] for r in recs])
        Smax = np.array([r[1] for r in recs])
        ax.plot(Ns, Smax, "o-", lw=1.3, ms=6, label=f"k_bulk={k}")
        # Reference: Page bound N/2
        ax.plot(Ns, Ns / 2, ":", lw=0.7, alpha=0.5,
                color=ax.lines[-1].get_color())
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N (boundary qubits)")
    ax.set_ylabel(r"$\max_L S(L)$ [bits]")
    ax.set_title(r"Peak entanglement vs $N$: tree code (solid) vs Page bound $N/2$ (dotted)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Phase 3b: tree code (HaPPY-like) -- area-law signature via QEC encoding",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = os.path.join(os.path.dirname(__file__) or ".",
                             "fig_A3a_phase3b_happy.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {fig_path}")
    print(f"Wrote results.json key: A3a_phase3b_happy_tree")


if __name__ == "__main__":
    main()
