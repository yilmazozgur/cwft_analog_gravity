"""
A3b -- HaPPY-like Clifford code with explicit emergent bulk geometry.

This is the framework's load-bearing post-Hayden-Preskill experiment. The
A3a tree-doubling code showed sub-Page entanglement on boundary regions
(the necessary condition for area-law). A3b tests the SUFFICIENT condition:
does the entanglement entropy of a boundary region match the substrate-
relative Ryu-Takayanagi prediction quantitatively?

SUBSTRATE: a balanced binary tree of qubits.
  - 2^(d+1) - 1 qubits total
  - Internal nodes (heap indices 0 .. 2^d - 2) = BULK qubits
  - Leaves (heap indices 2^d - 1 .. 2^(d+1) - 2) = BOUNDARY qubits
  - For each tree edge (parent, child), apply a random 2-qubit Clifford
  - Repeat for n_rounds rounds (uniform random Clifford entanglement)

The bulk geometry is the tree: bulk distance between two leaves = depth
of their lowest common ancestor; bulk min-cut between a leaf-subset A and
its complement = number of tree edges that separate them.

RT PREDICTION:  S(A) = eta_c * |gamma_A|
  where gamma_A is the minimum-cut surface in the bulk separating A from
  A-complement, and eta_c is a universal substrate-relative entropy
  density (the framework's substrate-relative analog of 1/(4 G_N hbar)).

For random Clifford on tree edges with n_rounds = 1, each edge contributes
~1 bit of entanglement, so we predict eta_c ~ 1. For multiple rounds, the
edge contribution may saturate.

MEASUREMENT:
  - Sweep tree depths d in {3, 4, 5, 6, 7} (N_leaves = 2^d in {8..128})
  - Multiple seeds per tree
  - Sample contiguous boundary subregions across all sizes L in {1..N-1},
    multiple starting positions per L
  - For each (region, seed, depth): record (L, |gamma_A|, S(A))

ANALYSIS:
  - Scatter S vs |gamma_A|
  - Test: linear fit slope universal across depths?
  - Test: subleading dependence on |A| beyond |gamma_A|?
  - Compare to Page bound min(L, N-L) for context

The chapter's Corollary 1.6.2 conjectures the substrate-relative RT formula
extends to substrates with explicit emergent bulk geometry. A3b tests this
on the simplest substrate where geometry is unambiguously defined.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stim

sys.path.insert(0, os.path.dirname(__file__) or ".")
from cwf_hp_lib import make_sim, stabilizer_matrix, entropy_region


# =========================================================================
# Tree topology
# =========================================================================

def tree_indices(depth):
    """Return (internal_indices, leaf_indices) for a balanced binary tree.

    Heap indexing: root at 0, children of v at 2v+1 and 2v+2.
    Internal: indices 0 .. 2^d - 2.
    Leaves: indices 2^d - 1 .. 2^(d+1) - 2.
    """
    N_total = (1 << (depth + 1)) - 1
    N_leaves = 1 << depth
    internal = list(range(N_total - N_leaves))
    leaves = list(range(N_total - N_leaves, N_total))
    return internal, leaves


def tree_edges(depth):
    """Return list of (parent, child) tree edges."""
    N_total = (1 << (depth + 1)) - 1
    edges = []
    for v in range((N_total - 1) // 2 + 1):  # all nodes that have at least one child
        for c in (2 * v + 1, 2 * v + 2):
            if c < N_total:
                edges.append((v, c))
    return edges


# =========================================================================
# Substrate construction
# =========================================================================

def build_tree_graph_state(depth, seed, measure_bulk=True):
    """Tree GRAPH STATE: |+>^N then CZ on each tree edge.

    This is the "perfect-tensor" analog of the random-Clifford tree code:
    each tree edge gets a deterministic CZ that produces a maximally
    entangled link (vs. random Clifford which entangles only ~80% of
    the time). For a graph state on a tree, after Z-measuring bulk
    qubits, the boundary entropy of region A equals exactly |gamma_A|
    (in bits). Predicted slope: eta_c = 1.0.

    This is the framework's analog of a deterministic HaPPY code; the
    random-Clifford version (build_tree_clifford_substrate) gives the
    "noisy" version with eta_c ~ 0.5.
    """
    internal, leaves = tree_indices(depth)
    N_total = len(internal) + len(leaves)
    edges = tree_edges(depth)
    sim = make_sim(N_total, seed=int(seed))

    # |+>^N
    for q in range(N_total):
        sim.h(int(q))
    # CZ on each tree edge (graph state)
    for (v, c) in edges:
        sim.cz(int(v), int(c))

    if measure_bulk:
        for q in internal:
            sim.measure(int(q))

    return sim, internal, leaves, edges


def build_tree_clifford_substrate(depth, seed, n_rounds=2,
                                    measure_bulk=True):
    """Construct the tree-Clifford substrate state.

    All qubits start in |0>. For n_rounds rounds, apply a random 2-qubit
    Clifford on each tree edge (in random order per round).

    If measure_bulk=True (default), after the Clifford layer measure
    every bulk qubit in the Z basis. This fixes the bulk state into a
    definite product (the "bulk vacuum" up to a random Z-basis
    classical outcome per seed). For a HaPPY-like code, the boundary
    state is then the codeword for the bulk-outcome, and the boundary
    entropy of a subregion is bounded by the cut-edge count in the
    bulk graph -- the Ryu-Takayanagi prediction.

    Returns: (sim, internal, leaves, edges)
    """
    internal, leaves = tree_indices(depth)
    N_total = len(internal) + len(leaves)
    edges = tree_edges(depth)

    sim = make_sim(N_total, seed=int(seed))
    rng = np.random.default_rng(int(seed) + 11111)

    for _ in range(n_rounds):
        order = list(edges)
        rng.shuffle(order)
        for (v, c) in order:
            tab = stim.Tableau.random(2)
            sim.do_tableau(tab, [int(v), int(c)])

    if measure_bulk:
        # Z-measure each bulk qubit. stim returns the outcome (random for
        # this Clifford state); we discard it. Post-measurement, bulk
        # qubits are in definite Z eigenstates; boundary entropy now
        # reflects only the cut-edge / code-distance structure.
        for q in internal:
            sim.measure(int(q))

    return sim, internal, leaves, edges


# =========================================================================
# Min-cut computation in a tree
# =========================================================================

def cut_size_for_leaf_subset(depth, A_leaf_positions):
    """Compute the minimum edge cut in a balanced binary tree separating
    leaves in A_leaf_positions from their complement.

    A_leaf_positions: iterable of indices in {0 .. 2^d - 1} (left-to-right
    positions of leaves; NOT the heap indices).

    Returns: integer cut size (minimum number of tree edges to remove so
    that every connected component has leaves of only one class).

    Algorithm: DP from leaves upward. For each node v and each color
    c in {A, Ac}, compute f[v][c] = the minimum number of cuts inside
    subtree(v) such that v's connected component is labeled c. At each
    child c, we either keep edge (v, c) (then c is in v's component,
    must match v's color) or cut edge (v, c) (cost +1, c's subtree
    becomes independent and is internally made homogeneous; min over
    both possible labels of c's component).

    Final answer: min(f[root][A], f[root][Ac]).
    """
    N_leaves = 1 << depth
    A = set(int(p) for p in A_leaf_positions)
    if not A or A == set(range(N_leaves)):
        return 0

    N_total = (1 << (depth + 1)) - 1
    INF = 10**9  # large enough sentinel
    # f[v] = [cost_for_A_color, cost_for_Ac_color]
    f = [[INF, INF] for _ in range(N_total)]
    # Leaves
    for i in range(N_leaves):
        heap_idx = (1 << depth) - 1 + i
        if i in A:
            f[heap_idx][0] = 0  # already A; cost 0
        else:
            f[heap_idx][1] = 0  # already Ac
    # Internal nodes, bottom-up
    for v in range((1 << depth) - 2, -1, -1):
        c_left = 2 * v + 1
        c_right = 2 * v + 2
        for color in (0, 1):
            cost = 0
            for c in (c_left, c_right):
                # Option (keep): edge (v, c) not cut. c is in v's component,
                # so c must be color. Cost contribution = f[c][color].
                # Option (cut): edge (v, c) is cut. Cost += 1. c's subtree
                # is independent, internally make it homogeneous of either
                # color: cost = min(f[c][0], f[c][1]).
                cost += min(f[c][color], 1 + min(f[c][0], f[c][1]))
            f[v][color] = cost
    return int(min(f[0][0], f[0][1]))


# =========================================================================
# Self-test: cut function correctness
# =========================================================================

def _selftest_cuts():
    """Sanity-check the min-cut function on cases with known answers.

    For a balanced binary tree, the min edge-cut between leaf-set A and
    its complement equals 1 plus the number of additional "homogeneous
    leaf clusters" beyond the first --- equivalently, you need 1 cut per
    distinct maximal-homogeneous subtree to peel off, minus 1 (the last
    piece is implicit). For contiguous regions and isolated leaves it
    works out as follows:
    """
    # depth=2 (4 leaves):
    # A = {0} (one corner leaf): min-cut = 1 (just cut the leaf's edge).
    assert cut_size_for_leaf_subset(2, {0}) == 1, \
        f"depth=2, A={{0}}: expected 1, got {cut_size_for_leaf_subset(2, {0})}"
    # A = {0, 1}: entirely-left subtree; cut edge (root, left-child) = 1.
    assert cut_size_for_leaf_subset(2, {0, 1}) == 1, \
        f"depth=2, A={{0,1}}: expected 1, got {cut_size_for_leaf_subset(2, {0, 1})}"
    # A = {0, 3}: opposite corners. Cut (left, left-leaf-right) + (right, right-leaf-left) = 2.
    assert cut_size_for_leaf_subset(2, {0, 3}) == 2, \
        f"depth=2, A={{0,3}}: expected 2, got {cut_size_for_leaf_subset(2, {0, 3})}"
    # A = {0, 2}: also two non-adjacent leaves => cut = 2.
    assert cut_size_for_leaf_subset(2, {0, 2}) == 2, \
        f"depth=2, A={{0,2}}: expected 2, got {cut_size_for_leaf_subset(2, {0, 2})}"
    # depth=3 (8 leaves):
    # A = full leaves => 0.
    assert cut_size_for_leaf_subset(3, set(range(8))) == 0
    # A = empty => 0.
    assert cut_size_for_leaf_subset(3, set()) == 0
    # A = left half {0..3}: one subtree, cut = 1.
    assert cut_size_for_leaf_subset(3, {0, 1, 2, 3}) == 1, \
        f"depth=3, left half: got {cut_size_for_leaf_subset(3, {0,1,2,3})}"
    # A = single leaf {0}: cut = 1.
    assert cut_size_for_leaf_subset(3, {0}) == 1, \
        f"depth=3, A={{0}}: got {cut_size_for_leaf_subset(3, {0})}"
    # A = two adjacent leaves {0, 1}: cut = 1 (they share a parent subtree).
    assert cut_size_for_leaf_subset(3, {0, 1}) == 1, \
        f"depth=3, A={{0,1}}: got {cut_size_for_leaf_subset(3, {0,1})}"
    # A = {0, 2}: not adjacent, share grandparent. min-cut depends on structure.
    # Tree at depth 3:
    #   root has left subtree (leaves 0-3) and right subtree (leaves 4-7).
    #   Left subtree: heap node 1, with children 3 (leaves 0, 1) and 4 (leaves 2, 3).
    #   A = {0, 2}: leaf 0 is in heap-3, leaf 2 is in heap-4.
    #   Both A-leaves are in the left subtree (heap-1). Within the left subtree:
    #     subtree(3) has one A, one Ac; subtree(4) has one A, one Ac. Both mixed.
    #     To separate {0, 2} from {1, 3}: cut leaves 1 and 3, OR... 2 cuts.
    #   No way to do it in fewer cuts (you have to peel off two leaves from two different subtrees).
    cut02 = cut_size_for_leaf_subset(3, {0, 2})
    print(f"  selftest: depth=3, A={{0,2}}: cut = {cut02} (expected 2)")
    assert cut02 == 2, f"depth=3, A={{0,2}}: expected 2, got {cut02}"
    print("  cut-function self-tests passed.")


# =========================================================================
# Measurement
# =========================================================================

def measure_S_of_region(sim, leaves, A_leaf_positions, N_total):
    """S(A) where A is the set of boundary qubits corresponding to
    leaf positions A_leaf_positions."""
    X, Z = stabilizer_matrix(sim, N_total)
    A_qubits = [leaves[int(p)] for p in A_leaf_positions]
    return entropy_region(X, Z, A_qubits, N_total)


# =========================================================================
# Sweep
# =========================================================================

def sample_contiguous_regions(N_leaves, n_positions_per_size):
    """Yield (L, start) for contiguous regions of size L starting at start.

    For each L in {1, ..., N-1}, sample n_positions_per_size starting
    positions uniformly across {0, ..., N-L}. Skip empty regions."""
    for L in range(1, N_leaves):
        n_starts = min(n_positions_per_size, N_leaves - L + 1)
        if n_starts == 1:
            yield L, 0
        else:
            step = (N_leaves - L) / (n_starts - 1) if n_starts > 1 else 0
            for k in range(n_starts):
                start = int(round(k * step))
                start = max(0, min(N_leaves - L, start))
                yield L, start


def run_one_seed(depth, seed, n_rounds, n_positions_per_size):
    """Build the tree-Clifford state, sweep regions, return list of dicts."""
    sim, internal, leaves, edges = build_tree_clifford_substrate(
        depth, seed=seed, n_rounds=n_rounds)
    N_total = len(internal) + len(leaves)
    N_leaves = len(leaves)

    # Cache the stabilizer matrix once
    X, Z = stabilizer_matrix(sim, N_total)

    results = []
    seen_regions = set()
    for L, start in sample_contiguous_regions(N_leaves, n_positions_per_size):
        positions = tuple(range(start, start + L))
        if positions in seen_regions:
            continue
        seen_regions.add(positions)
        A_pos = set(positions)
        cut = cut_size_for_leaf_subset(depth, A_pos)
        # entropy via stabilizer formula
        A_qubits = [leaves[p] for p in positions]
        S = entropy_region(X, Z, A_qubits, N_total)
        results.append(dict(L=L, start=start, cut=cut, S=float(S)))
    return results


def main():
    _selftest_cuts()
    print()

    depths = [3, 4, 5, 6, 7]
    n_rounds = 2
    n_seeds = 12
    n_positions_per_size = 4

    all_results = {}
    t0 = time.time()
    print(f"A3b: HaPPY-like tree code -- emergent-geometry RT test")
    print(f"  depths: {depths}")
    print(f"  n_rounds={n_rounds}, n_seeds={n_seeds}, "
          f"n_positions_per_size={n_positions_per_size}\n")

    for d in depths:
        N_leaves = 1 << d
        N_total = (1 << (d + 1)) - 1
        t_d = time.time()
        per_seed = []
        for s in range(n_seeds):
            rs = run_one_seed(d, seed=10 + s,
                              n_rounds=n_rounds,
                              n_positions_per_size=n_positions_per_size)
            per_seed.append(rs)
        # aggregate
        flat = [r for s_list in per_seed for r in s_list]
        n_pts = len(flat)
        all_results[f"d={d}"] = dict(
            depth=d, N_leaves=N_leaves, N_total=N_total,
            n_rounds=n_rounds, n_seeds=n_seeds,
            points=flat,
            elapsed_s=time.time() - t_d,
        )
        # quick fit slope
        cuts = np.array([r["cut"] for r in flat], float)
        Svals = np.array([r["S"] for r in flat], float)
        # restrict to cuts > 0 to avoid the empty-region case
        mask = cuts > 0
        if mask.any():
            slope = float(np.sum(Svals[mask] * cuts[mask]) /
                          np.sum(cuts[mask] ** 2))
            r2 = float(1 - np.sum((Svals[mask] - slope * cuts[mask]) ** 2) /
                       np.sum((Svals[mask] - Svals[mask].mean()) ** 2 + 1e-12))
        else:
            slope = float("nan"); r2 = float("nan")
        all_results[f"d={d}"]["slope"] = slope
        all_results[f"d={d}"]["R2"] = r2
        dt_d = time.time() - t_d
        print(f"  d={d}  N={N_leaves:3d}  N_total={N_total:3d}  "
              f"n_pts={n_pts:5d}  slope eta_c={slope:.3f}  R^2={r2:+.3f}  "
              f"({dt_d:.1f}s)")

    print(f"\nTotal runtime: {time.time() - t0:.0f}s")

    # Save
    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["A3b_emergent_geometry"] = dict(
        depths=depths, n_rounds=n_rounds, n_seeds=n_seeds,
        n_positions_per_size=n_positions_per_size,
        by_depth=all_results,
    )
    json.dump(r_all, open(out, "w"), indent=2)

    # Plot
    plot_results(all_results, depths)
    print(f"Wrote results.json key: A3b_emergent_geometry")


def plot_results(all_results, depths):
    """Headline plot: S vs cut size, color-coded by tree depth."""
    fig = plt.figure(figsize=(15.5, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30)
    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_scatter_zoom = fig.add_subplot(gs[0, 1])
    ax_slope = fig.add_subplot(gs[0, 2])
    ax_page = fig.add_subplot(gs[1, 0])
    ax_residual = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[1, 2])

    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.85, len(depths)))

    # (a) Scatter S vs cut size
    for d, col in zip(depths, colors):
        rec = all_results[f"d={d}"]
        cuts = np.array([r["cut"] for r in rec["points"]], float)
        Svals = np.array([r["S"] for r in rec["points"]], float)
        ax_scatter.scatter(cuts, Svals, s=12, alpha=0.35, color=col,
                            label=f"d={d} (N={rec['N_leaves']})")
    # universal-slope reference: slope 1 line
    xrange = np.linspace(0, max(np.array([r["cut"] for r in all_results[f"d={depths[-1]}"]["points"]]).max(), 8), 60)
    ax_scatter.plot(xrange, xrange, "k:", lw=0.8, alpha=0.5, label=r"slope $=1$")
    ax_scatter.set_xlabel(r"cut size $|\gamma_A|$ (tree edges)")
    ax_scatter.set_ylabel(r"entanglement entropy $S(A)$ [bits]")
    ax_scatter.set_title(r"(a) $S(A)$ vs $|\gamma_A|$ across tree depths")
    ax_scatter.legend(fontsize=8)
    ax_scatter.grid(alpha=0.3)

    # (b) Zoom in low cut
    for d, col in zip(depths, colors):
        rec = all_results[f"d={d}"]
        cuts = np.array([r["cut"] for r in rec["points"]], float)
        Svals = np.array([r["S"] for r in rec["points"]], float)
        m = cuts <= 10
        ax_scatter_zoom.scatter(cuts[m], Svals[m], s=18, alpha=0.4,
                                 color=col, label=f"d={d}")
    ax_scatter_zoom.plot([0, 10], [0, 10], "k:", lw=0.8, alpha=0.5,
                          label=r"slope $=1$")
    ax_scatter_zoom.set_xlabel(r"$|\gamma_A|$ (zoomed $\leq 10$)")
    ax_scatter_zoom.set_ylabel(r"$S(A)$")
    ax_scatter_zoom.set_title("(b) Same, zoomed to small cuts")
    ax_scatter_zoom.legend(fontsize=8)
    ax_scatter_zoom.grid(alpha=0.3)

    # (c) Per-depth slope
    slopes = [all_results[f"d={d}"]["slope"] for d in depths]
    r2s = [all_results[f"d={d}"]["R2"] for d in depths]
    ax_slope.plot(depths, slopes, "o-", lw=1.5, ms=7, color="C0",
                   label=r"$\eta_c$ (slope)")
    ax_slope.axhline(1.0, color="k", ls=":", lw=0.6, alpha=0.5)
    ax_slope.set_xlabel("tree depth $d$")
    ax_slope.set_ylabel(r"$\eta_c$ (fit slope of $S = \eta_c \, |\gamma_A|$)")
    ax_slope.set_title(r"(c) $\eta_c$ across tree depths (universal?)")
    ax_slope.set_ylim(0, max(1.4, max(slopes) * 1.1))
    ax_slope.grid(alpha=0.3)
    ax_slope.legend(fontsize=8)
    # second y axis: R^2
    ax2 = ax_slope.twinx()
    ax2.plot(depths, r2s, "s--", lw=1.0, ms=6, color="C3", alpha=0.6,
             label=r"$R^2$")
    ax2.set_ylabel(r"$R^2$ of linear fit", color="C3")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="C3")

    # (d) Page bound comparison
    for d, col in zip(depths, colors):
        rec = all_results[f"d={d}"]
        N = rec["N_leaves"]
        Ls = np.array([r["L"] for r in rec["points"]], float)
        Svals = np.array([r["S"] for r in rec["points"]], float)
        # group by L, take mean
        unique_L = np.unique(Ls.astype(int))
        S_mean = np.array([Svals[Ls == ll].mean() for ll in unique_L])
        ax_page.plot(unique_L / N, S_mean / (N / 2), "o-", color=col,
                      lw=1.3, ms=4, label=f"d={d}")
    # Page bound min(L, N-L)
    xs = np.linspace(0, 1, 100)
    ax_page.plot(xs, 2 * np.minimum(xs, 1 - xs), "k:", lw=0.8, alpha=0.5,
                  label="Page bound")
    ax_page.set_xlabel("$L/N$")
    ax_page.set_ylabel(r"$S(L)/(N/2)$")
    ax_page.set_title("(d) Mean $S$ vs $L$: tree code (color) vs Page bound (dotted)")
    ax_page.legend(fontsize=8)
    ax_page.grid(alpha=0.3)

    # (e) Residual analysis: S - eta_c * cut, per region size
    for d, col in zip(depths, colors):
        rec = all_results[f"d={d}"]
        slope = rec["slope"]
        cuts = np.array([r["cut"] for r in rec["points"]], float)
        Svals = np.array([r["S"] for r in rec["points"]], float)
        Ls = np.array([r["L"] for r in rec["points"]], float)
        N = rec["N_leaves"]
        residual = Svals - slope * cuts
        # plot residual vs L/N
        ax_residual.scatter(Ls / N, residual, s=10, alpha=0.4, color=col,
                              label=f"d={d}")
    ax_residual.axhline(0, color="k", ls=":", lw=0.6)
    ax_residual.set_xlabel("$L/N$")
    ax_residual.set_ylabel(r"$S - \eta_c|\gamma_A|$")
    ax_residual.set_title("(e) Residual from linear RT prediction")
    ax_residual.legend(fontsize=8)
    ax_residual.grid(alpha=0.3)

    # (f) summary text
    ax_summary.axis("off")
    summary = "Headline result (A3b):\n\n"
    for d in depths:
        rec = all_results[f"d={d}"]
        summary += (f"  d={d}  N={rec['N_leaves']:3d}  N_tot={rec['N_total']:3d}  "
                    f"$\\eta_c$={rec['slope']:.3f}  $R^2$={rec['R2']:+.2f}\n")
    slopes_arr = np.array([all_results[f"d={d}"]["slope"] for d in depths])
    summary += (f"\n  Slope mean: {slopes_arr.mean():.3f}\n"
                f"  Slope CV  : {slopes_arr.std() / slopes_arr.mean():.3f}\n"
                f"  Slope range: [{slopes_arr.min():.3f}, {slopes_arr.max():.3f}]\n")
    if slopes_arr.std() / slopes_arr.mean() < 0.10:
        summary += "\n  -> UNIVERSAL $\\eta_c$ (CV<10%) -- RT formula holds!"
    elif slopes_arr.std() / slopes_arr.mean() < 0.25:
        summary += "\n  -> Approximately universal (CV<25%)"
    else:
        summary += "\n  -> NOT universal; RT formula fails"
    ax_summary.text(0.02, 0.98, summary, transform=ax_summary.transAxes,
                     fontsize=9, family="monospace", verticalalignment="top")

    fig.suptitle("A3b: HaPPY-like tree code -- emergent-geometry Ryu-Takayanagi test",
                 fontsize=12)
    fig_path = os.path.join(os.path.dirname(__file__) or ".",
                             "fig_A3b_emergent_geometry.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
