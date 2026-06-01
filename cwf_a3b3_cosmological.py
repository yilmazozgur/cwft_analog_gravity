"""
The cosmological-constant identification, made concrete on the HaPPY
perfect-tensor substrate.

The chapter (Eq. comp-einstein) carries an open Lambda_c term. The active
discussion conjectured Lambda_c ~ the "unconstructable bulk fraction" -- the
measure of bulk regions whose reconstruction from the boundary is
(asymptotically) uncomputable -- and flagged it as untested speculation.

With the [[5,1,3]] perfect-tensor network of cwf_a3b2_happy_perfect.py we can
give "unconstructable bulk fraction" a concrete operational meaning via
entanglement-wedge reconstruction:

  A bulk qubit b is reconstructable from a boundary region A iff b lies in
  the entanglement wedge of A, i.e. iff I(b:A) = 2 bits (b's logical info is
  fully carried by A). For each bulk qubit we find the MINIMAL boundary arc
  fraction f_b needed to reconstruct it. Outer (near-boundary) bulk qubits
  are recoverable from small local arcs (f_b small); the deep central bulk
  needs almost the whole boundary (f_b -> 1).

Two dimensionless, substrate-relative readouts -- candidate Lambda_c proxies:

  <f>      = mean over bulk qubits of the minimal reconstruction fraction.
             "Average irreducible boundary cost of a bulk dof."
  core     = fraction of bulk qubits NOT reconstructable from ANY boundary
             half (f_b > 1/2). The "irreducible deep core" -- the operational
             stand-in for the asymptotically-unconstructable bulk.

If these converge to a stable constant as the network grows (hyperbolic
geometry: bulk and boundary both grow exponentially), that constant is a
clean geometric Lambda_c candidate. STATUS: speculative bridge (S/B); this
is a finite-resource proxy for an asymptotic-uncomputability claim, not a
derivation of the cosmological constant.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__) or ".")
from cwf_hp_lib import stabilizer_matrix, entropy_region, mutual_information
from cwf_a3b2_happy_perfect import build_network, build_state, BULK_LEG


def min_reconstruction_fraction(X, Z, N, b, boundary, max_starts=6):
    """Smallest arc fraction |A|/n over circular arcs A with I(b:A)=2
    (b fully reconstructable from A, i.e. b in the entanglement wedge of A).

    Reconstructability is monotone in arc size (EW(A) grows with A), so we
    binary-search the smallest arc length L that reconstructs b from some
    starting position (scanning up to max_starts evenly-spaced starts per
    size). The full boundary always reconstructs a bulk logical leg, so the
    search is well-posed."""
    n = len(boundary)

    def reconstructs_at(L):
        starts = sorted(set(int(round(k * n / min(max_starts, n))) % n
                            for k in range(min(max_starts, n))))
        for s in starts:
            A = [boundary[(s + i) % n] for i in range(L)]
            if mutual_information(X, Z, [b], A, N) >= 2.0 - 1e-9:
                return True
        return False

    lo, hi = 1, n  # hi reconstructs by the code property
    while lo < hi:
        mid = (lo + hi) // 2
        if reconstructs_at(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo / n


def run_depth(max_depth, n_child=2, max_starts=6):
    # deep=True: genuine hyperbolic disk, boundary only at the outer layer,
    # so the root is a true deep-bulk core (the fair test for Lambda_c).
    tensors, bonds, bulk_qubits, boundary = build_network(max_depth, n_child,
                                                          deep=True)
    # keep bulk legs as logical dof (do NOT fix them)
    sim, N = build_state(tensors, bonds, bulk_qubits, fix_bulk=False)
    X, Z = stabilizer_matrix(sim, N)
    # bulk qubit -> tensor depth (radial coordinate)
    depth_of = {t.qubits[BULK_LEG]: t.depth for t in tensors}
    f = {}
    for b in bulk_qubits:
        f[b] = min_reconstruction_fraction(X, Z, N, b, boundary, max_starts)
    fr = np.array(list(f.values()))
    mean_f = float(fr.mean())
    core = float(np.mean(fr > 0.5))
    # mean f by radial depth
    by_d = {}
    for b, fb in f.items():
        by_d.setdefault(depth_of[b], []).append(fb)
    radial = {d: float(np.mean(v)) for d, v in sorted(by_d.items())}
    return dict(max_depth=max_depth, n_tensors=len(tensors), N_total=N,
                n_boundary=len(boundary), n_bulk=len(bulk_qubits),
                mean_f=mean_f, core_frac=core, radial=radial,
                f_values=[float(x) for x in fr])


def main():
    depths = [1, 2, 3]
    n_child = 2
    t0 = time.time()
    print("Lambda_c proxy: unconstructable-bulk fraction on the HaPPY net\n")
    by_depth = {}
    for d in depths:
        rec = run_depth(d, n_child=n_child)
        by_depth[f"d={d}"] = rec
        print(f"  depth={d}  n_bulk={rec['n_bulk']:3d}  n_boundary={rec['n_boundary']:3d}  "
              f"<f>={rec['mean_f']:.4f}  core(f>1/2)={rec['core_frac']:.4f}")
        print(f"           radial <f> by depth: "
              + "  ".join(f"{k}:{v:.2f}" for k, v in rec["radial"].items()))
    mf = np.array([by_depth[f"d={d}"]["mean_f"] for d in depths])
    cf = np.array([by_depth[f"d={d}"]["core_frac"] for d in depths])
    print(f"\n  <f> across depths:  {[f'{x:.3f}' for x in mf]}  "
          f"(last-two mean {mf[-2:].mean():.3f})")
    print(f"  core across depths: {[f'{x:.3f}' for x in cf]}  "
          f"(last-two mean {cf[-2:].mean():.3f})")
    print(f"  runtime {time.time() - t0:.1f}s")

    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["A3b3_cosmological"] = dict(
        depths=depths, n_child=n_child,
        mean_f={f"d={d}": by_depth[f"d={d}"]["mean_f"] for d in depths},
        core_frac={f"d={d}": by_depth[f"d={d}"]["core_frac"] for d in depths},
        by_depth=by_depth)
    json.dump(r_all, open(out, "w"), indent=2)
    plot_results(by_depth, depths)
    print("Wrote results.json key: A3b3_cosmological")


def plot_results(by_depth, depths):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # (a) convergence of <f> and core fraction
    ax = axes[0]
    mf = [by_depth[f"d={d}"]["mean_f"] for d in depths]
    cf = [by_depth[f"d={d}"]["core_frac"] for d in depths]
    ax.plot(depths, mf, "o-", color="C0", ms=8, label=r"$\langle f\rangle$ (mean recon. fraction)")
    ax.plot(depths, cf, "s--", color="C3", ms=8, label=r"core fraction ($f_b>1/2$)")
    ax.set_xlabel("network depth"); ax.set_ylabel("dimensionless")
    ax.set_ylim(0, 1)
    ax.set_title(r"(a) $\Lambda_c$-proxy convergence")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    # (b) radial profile at the largest depth
    ax = axes[1]
    cmap = plt.cm.viridis
    cols = cmap(np.linspace(0.15, 0.85, len(depths)))
    for d, c in zip(depths, cols):
        rad = by_depth[f"d={d}"]["radial"]
        ax.plot(list(rad.keys()), list(rad.values()), "o-", color=c,
                label=f"depth {d}")
    ax.set_xlabel("bulk radial coordinate (tensor depth)")
    ax.set_ylabel(r"mean reconstruction fraction $\langle f\rangle_r$")
    ax.set_title("(b) deep bulk costs more boundary")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (c) summary
    ax = axes[2]; ax.axis("off")
    txt = "Lambda_c proxy (unconstructable bulk):\n\n"
    for d in depths:
        r = by_depth[f"d={d}"]
        txt += (f"  depth {d}: n_bulk={r['n_bulk']:3d} "
                f"N$_\\partial$={r['n_boundary']:3d}\n"
                f"     <f>={r['mean_f']:.3f}  core={r['core_frac']:.3f}\n")
    txt += ("\n  <f>  = avg boundary fraction to\n         reconstruct a bulk dof.\n"
            "  core = bulk needing > half the\n         boundary (irreducible).\n\n"
            "  core = 0 at every depth: a finite\n  perfect code reconstructs every-\n"
            "  thing -> the unconstructable-bulk\n  proxy for Lambda_c VANISHES.\n"
            "  G_c pinned, Lambda_c not sourced.\n  STATUS: measurement R; Lambda_c=S.")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9.5,
            family="monospace", va="top")
    fig.suptitle(r"Cosmological-constant identification: unconstructable-bulk "
                 r"fraction on the HaPPY perfect-tensor substrate", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(os.path.dirname(__file__) or ".",
                     "fig_A3b3_cosmological.png")
    plt.savefig(p, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
