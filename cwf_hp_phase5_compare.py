"""
A3a Phase 5 -- cross-substrate aggregation and final figures.

Reads results.json keys produced by phases 1-4 and 3b, and produces:
  1. A unified scrambling-time figure.
  2. A unified Page-curve / HP-recovery summary.
  3. The headline area-law vs volume-law comparison: random scrambling
     (volume-law / Page bound) vs tree-code (area-law / sub-Page).
  4. A summary table (substrate x diagnostic).

This is the chapter-ready figure / table set.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    here = os.path.dirname(__file__) or "."
    out = os.path.join(here, "results.json")
    res = json.load(open(out))

    p1 = res.get("A3a_phase1_hp_decoding", {})
    p2 = res.get("A3a_phase2_pagecurves", {})
    p3b = res.get("A3a_phase3b_happy_tree", {})
    p4 = res.get("A3a_phase4_scrambling_time", {})

    fig = plt.figure(figsize=(15.5, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.32)

    # ---- (a) scrambling time scaling (phase 4) ----
    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.tab10
    if p4 and "by_substrate" in p4:
        for i, (sub, recs) in enumerate(p4["by_substrate"].items()):
            Ns = [r["N"] for r in recs]
            ts = [r["t_star_med"] for r in recs]
            tstd = [r["t_star_std"] for r in recs]
            ax.errorbar(Ns, ts, yerr=tstd, fmt="o-", color=cmap(i),
                         lw=1.4, ms=5, capsize=3, label=sub)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N (qubits)")
        ax.set_ylabel(r"$t_*$ (steps)")
        ax.set_title("(a) Scrambling time $t_*(N)$")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)

    # ---- (b) HP recovery curve (phase 1) for k=2 across substrates at N~64 ----
    ax = fig.add_subplot(gs[0, 1])
    if p1 and "by_substrate" in p1:
        target_N = 64
        target_k = 2
        for i, (sub, sub_results) in enumerate(p1["by_substrate"].items()):
            best_match = None
            for key, rec in sub_results.items():
                if rec["k"] == target_k and abs(rec["N"] - target_N) <= 2:
                    if best_match is None or abs(rec["N"] - target_N) < abs(best_match["N"] - target_N):
                        best_match = rec
            if best_match is None: continue
            xs = np.array(best_match["n_revealed"]) / best_match["N"]
            Im = np.array(best_match["I_mean"]) / (2 * best_match["k"])
            ax.plot(xs, Im, "-", color=cmap(i), lw=1.4,
                    label=f"{sub} (N={best_match['N']})")
        ax.axhline(1.0, color="k", ls=":", lw=0.6, alpha=0.5)
        ax.axvline(0.5, color="r", ls="--", lw=0.6, alpha=0.4, label="Page time")
        ax.set_xlabel("$|L|/N$ (fraction revealed)")
        ax.set_ylabel("$I(R, L)/2k$")
        ax.set_title(f"(b) HP recovery at $N \\approx {target_N}$, $k={target_k}$")
        ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # ---- (c) Page-curves overlay (phase 2) at N~64 ----
    ax = fig.add_subplot(gs[0, 2])
    if p2 and "by_substrate" in p2:
        target_N = 64
        for i, (sub, sub_results) in enumerate(p2["by_substrate"].items()):
            best_match = None
            for key, rec in sub_results.items():
                if abs(rec["N"] - target_N) <= 2:
                    if best_match is None or abs(rec["N"] - target_N) < abs(best_match["N"] - target_N):
                        best_match = rec
            if best_match is None: continue
            N = best_match["N"]
            S = np.array(best_match["S_mean"])
            Ls = np.arange(N + 1)
            ax.plot(Ls / N, S / (N / 2), "-", color=cmap(i), lw=1.4,
                    label=f"{sub} (N={N})")
        ax.axhline(1.0, color="k", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlabel("$L/N$")
        ax.set_ylabel("$S(L)/(N/2)$")
        ax.set_title(f"(c) Page curves at $N \\approx {target_N}$")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.15)
        ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # ---- (d, e) headline area-law-vs-volume-law (phase 3b) ----
    ax = fig.add_subplot(gs[1, :2])
    if p3b and "tree_configs" in p3b:
        cmap2 = plt.cm.viridis
        configs = list(p3b["tree_configs"].items())
        colors = cmap2(np.linspace(0.05, 0.85, len(configs)))
        for (label, rec), col in zip(configs, colors):
            N = rec["N"]
            S = np.array(rec["S_mean"])
            Ls = np.arange(N + 1)
            ax.plot(Ls / N, S, "-", color=col, lw=1.4, label=label)
        # overlay Page curves
        if "random_clifford_baseline" in p3b:
            for key, rec in p3b["random_clifford_baseline"].items():
                N = rec["N"]
                Ls = np.arange(N + 1)
                page = np.minimum(Ls, N - Ls)
                ax.plot(Ls / N, page, "k:", lw=0.7, alpha=0.5)
        ax.set_xlabel("$L/N$ (contiguous boundary region fraction)")
        ax.set_ylabel("$S(L)$ [bits]")
        ax.set_title("(d) Headline: tree-code $S(L)$ (color) vs Page-bound $\\min(L, N-L)$ (dotted)\n"
                     "Tree code sits well below the Page bound: an empirical area-law signature.")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    # ---- (f) max-S vs N (the headline scaling) ----
    ax = fig.add_subplot(gs[1, 2])
    if p3b and "tree_configs" in p3b:
        by_k = {}
        for label, rec in p3b["tree_configs"].items():
            by_k.setdefault(rec["k_bulk"], []).append((rec["N"], rec["max_S"]))
        for k, pts in sorted(by_k.items()):
            pts = sorted(pts)
            Ns = [p[0] for p in pts]; Ss = [p[1] for p in pts]
            ax.plot(Ns, Ss, "o-", lw=1.4, ms=6, label=f"tree k_bulk={k}")
        # reference Page bound N/2
        all_N = sorted(set([rec["N"] for rec in p3b["tree_configs"].values()]))
        ax.plot(all_N, [n / 2 for n in all_N], "k--", lw=1.0, alpha=0.6,
                label="Page bound N/2")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("N (boundary qubits)")
        ax.set_ylabel(r"$\max_L S(L)$")
        ax.set_title("(e) Max entanglement vs N: tree-code stays $\\ll$ Page bound")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

    # ---- (g) summary table ----
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    table_text = (
        "PHASE 1 (HP recovery):    All 4 generic substrates achieve full recovery $I(R,L)=2k$ near $|L|/N \\approx 0.5$, matching the\n"
        "                          Page-time prediction. Confirms information IS recoverable from radiation -- ENCODING-BASED HORIZON.\n\n"
        "PHASE 2 (Page curves):    All 4 generic substrates produce textbook Page curves $S(L) = \\min(L, N-L)$ when fully scrambled.\n"
        "                          Confirms full scrambling capacity across substrate architectures.\n\n"
        "PHASE 3b (tree code):     Tree-doubling Clifford code shows $S(L) \\ll \\min(L, N-L)$. Max-S grows much more slowly than $N/2$\n"
        "                          as N grows. EMPIRICAL AREA-LAW: an encoding-based substrate class WITH sub-Page entanglement.\n\n"
        "PHASE 4 (scrambling t*): All-to-all and MERA-tree scale ~log N (fast scramblers); 2D brick-wall ~ sqrt(N) (local 2D);\n"
        "                          power-law sits between depending on alpha.\n\n"
        "STRUCTURAL CONCLUSION: Generic scrambling substrates conserve information (HP recovery works) but have volume-law entropy.\n"
        "    The tree-code (HaPPY-like) substrate has BOTH information conservation AND area-law entropy. This is the substrate class\n"
        "    that the chapter's Corollary 1.6.2 / Conjecture 1.6.3 identifies as the only candidate for computational Einstein dynamics."
    )
    ax.text(0.01, 0.97, table_text, transform=ax.transAxes,
            fontsize=9, family="monospace", verticalalignment="top")

    fig.suptitle("A3a -- Hayden-Preskill experiment, cross-substrate summary",
                 fontsize=12, y=0.995)
    fig_path = os.path.join(here, "fig_A3a_phase5_compare.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
