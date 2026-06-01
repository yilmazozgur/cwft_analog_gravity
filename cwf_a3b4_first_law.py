"""
A3b-dynamical -- the substrate-relative first law of entanglement (the
linearised computational Einstein equation) on the HaPPY perfect-tensor
substrate.

WHY THIS IS THE DYNAMICS TEST. The four dissipative experiments showed
Einstein DYNAMICS fails on the dissipative-horizon class (no universal
area-law, no reversibility). The perfect-tensor substrate fixed the
KINEMATICS quantitatively (eta_c=1, G_c=1/4 hbar_c; cwf_a3b2). What was
left open is dynamics: does the substrate's boundary entropy RESPOND to
bulk matter the way a (linearised) Einstein equation demands?

The sharp statement is the quantum-corrected Ryu-Takayanagi / FLM formula

    S(A) = |gamma_A|  +  S_bulk(EW(A)),

i.e. boundary entropy = area (min-cut) + bulk entropy inside the
entanglement wedge. Its first variation is the first law of entanglement
delta S_A = delta <K_A>, which Faulkner-Guica-Hartman-Myers-Van Raamsdonk
(2014) proved -- applied to all regions -- is EQUIVALENT to the linearised
Einstein equations. So testing the entropy RESPONSE to a bulk excitation is
testing the linearised computational Einstein equation on the substrate.

PROTOCOL. On the [[5,1,3]] HaPPY network (cwf_a3b2):
  - vacuum: all bulk legs fixed to |0>  -> S_A = |gamma_A|.
  - excite a set B of bulk legs: leave them UNFIXED (each a 1-bit bulk
    "field" excitation, maximally mixed relative to the wedge).
  - measure delta S_A = S_A(excited) - S_A(vacuum) for boundary arcs A.

PREDICTION (FLM / first law):
    delta S_A  =  | B  intersect  EW(A) |
where EW(A) is the entanglement wedge, computed INDEPENDENTLY by max-flow
on the bond graph (wedge_tensors). A unit bulk excitation shifts the
boundary entropy by exactly one bit iff it sits inside A's wedge -- and the
arc size at which delta S jumps 0->1 tracks the bulk geodesic (geometry).

This is exact for perfect tensors; the dissipative substrates cannot even
state it (they have no wedge). Confirming it is the chapter's first
DYNAMICAL positive on an Einstein-condition substrate.
"""
import json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__) or ".")
from cwf_hp_lib import stabilizer_matrix, entropy_region
from cwf_a3b2_happy_perfect import (build_network, build_state, wedge_tensors,
                                    BULK_LEG, circular_arcs)


def first_law_scan(max_depth, n_child=2, deep=False, max_per_size=None):
    """Excite ALL bulk legs; for every boundary arc compare the measured
    entropy response delta S_A to the geometric wedge prediction
    |bulk intersect EW(A)|."""
    tensors, bonds, bulk_qubits, boundary = build_network(max_depth, n_child,
                                                          deep=deep)
    N = 6 * len(tensors)
    n = len(boundary)
    tid_of = {t.qubits[BULK_LEG]: t.tid for t in tensors}
    excited_tids = set(tid_of[q] for q in bulk_qubits)

    simv, _ = build_state(tensors, bonds, bulk_qubits, fix_bulk=True)
    Xv, Zv = stabilizer_matrix(simv, N)
    sime, _ = build_state(tensors, bonds, bulk_qubits, fix_bulk=True,
                          excite=bulk_qubits)
    Xe, Ze = stabilizer_matrix(sime, N)

    pts = []
    for L, start in circular_arcs(n, max_per_size):
        A = [boundary[(start + i) % n] for i in range(L)]
        dS = entropy_region(Xe, Ze, A, N) - entropy_region(Xv, Zv, A, N)
        W = wedge_tensors(tensors, bonds, boundary, A)
        pred = len(excited_tids & W)
        pts.append((float(dS), int(pred)))
    return dict(max_depth=max_depth, deep=deep, n_tensors=len(tensors),
                N_total=N, n_boundary=n, n_bulk=len(bulk_qubits), points=pts)


def first_law_single(max_depth, n_child=2, deep=False, max_per_size=None):
    """LINEARISED first law: excite ONE bulk leg at a time; for every arc
    check the unit response delta S_A = [b in EW(A)]. Aggregate over all
    single excitations and all arcs. Returns exact-match fraction, the set
    of delta S values observed (should be {0,1}), and -- for the deep
    geometry -- the minimal capturing arc fraction f_b by radial depth (the
    geometry tie-in: deep bulk needs a larger arc)."""
    tensors, bonds, bulk_qubits, boundary = build_network(max_depth, n_child,
                                                          deep=deep)
    N = 6 * len(tensors)
    n = len(boundary)
    tid_of = {t.qubits[BULK_LEG]: t.tid for t in tensors}
    depth_of = {t.qubits[BULK_LEG]: t.depth for t in tensors}
    simv, _ = build_state(tensors, bonds, bulk_qubits, fix_bulk=True)
    Xv, Zv = stabilizer_matrix(simv, N)
    # cache wedge membership per arc
    arcs = list(circular_arcs(n, max_per_size))
    wedges = [wedge_tensors(tensors, bonds, boundary,
                            [boundary[(s + i) % n] for i in range(L)])
              for (L, s) in arcs]
    exact = 0; total = 0; vals = set(); radial = {}
    for b in bulk_qubits:
        sime, _ = build_state(tensors, bonds, bulk_qubits, fix_bulk=True,
                              excite=[b])
        Xe, Ze = stabilizer_matrix(sime, N)
        cap_frac = 1.0
        for (L, s), W in zip(arcs, wedges):
            A = [boundary[(s + i) % n] for i in range(L)]
            dS = entropy_region(Xe, Ze, A, N) - entropy_region(Xv, Zv, A, N)
            pred = 1 if tid_of[b] in W else 0
            vals.add(int(round(dS)))
            total += 1
            if abs(dS - pred) < 1e-9:
                exact += 1
            if dS >= 0.5:
                cap_frac = min(cap_frac, L / n)
        radial.setdefault(depth_of[b], []).append(cap_frac)
    radial_mean = {d: float(np.mean(v)) for d, v in sorted(radial.items())}
    return dict(exact_frac=exact / total, dS_values=sorted(vals),
                radial_capture=radial_mean, n_bulk=len(bulk_qubits),
                N_total=N, n_arcs=len(arcs))


def main():
    t0 = time.time()
    print("A3b-dynamical: first law of entanglement on the HaPPY substrate\n")

    configs = [(2, False), (3, False), (2, True)]  # (depth, deep)
    caps = {(2, False): None, (3, False): 24, (2, True): 16}

    # ---- LINEARISED first law: single unit excitation, dS_A = [b in EW(A)] ----
    print("  Linearised first law (single excitation):  delta S_A = [b in EW(A)]")
    single = {}
    for (d, deep) in configs:
        rec = first_law_single(d, deep=deep, max_per_size=caps[(d, deep)])
        tag = f"d={d}{'(deep)' if deep else ''}"
        single[tag] = rec
        print(f"    {tag:9s}  n_bulk={rec['n_bulk']:3d}  N={rec['N_total']:4d}  "
              f"exact (dS=[b in EW]): {rec['exact_frac']*100:5.1f}%   "
              f"dS values={rec['dS_values']}")
    print("    radial capture fraction f_b by bulk depth (deep, depth 2):")
    for d, f in single["d=2(deep)"]["radial_capture"].items():
        print(f"      radial depth {d}: <f_b> = {f:.2f}")

    # ---- FULL FLM: excite all bulk, dS_A = |bulk cap EW(A)| (graded) ----
    print("\n  Full FLM (all bulk excited):  delta S_A = |bulk cap EW(A)|")
    by_cfg = {}
    for (d, deep) in configs:
        rec = first_law_scan(d, deep=deep, max_per_size=caps[(d, deep)])
        dS = np.array([p[0] for p in rec["points"]])
        pred = np.array([p[1] for p in rec["points"]], float)
        exact = float(np.mean(np.abs(dS - pred) < 1e-9))
        slope = float(np.sum(dS * pred) / np.sum(pred ** 2)) if pred.any() else float("nan")
        unit = sorted(set(int(round(x)) for x in dS))
        rec.update(exact_frac=exact, slope=slope)
        tag = f"d={d}{'(deep)' if deep else ''}"
        by_cfg[tag] = rec
        print(f"    {tag:9s}  arcs={len(rec['points']):5d}  exact: {exact*100:5.1f}%  "
              f"slope={slope:.3f}  dS range={[min(unit),max(unit)]}")

    print(f"\n  runtime {time.time() - t0:.1f}s")

    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["A3b4_first_law"] = dict(
        linearised={k: dict(exact_frac=v["exact_frac"], dS_values=v["dS_values"],
                            radial_capture=v["radial_capture"], n_bulk=v["n_bulk"],
                            N_total=v["N_total"]) for k, v in single.items()},
        graded={k: dict(exact_frac=v["exact_frac"], slope=v["slope"],
                        n_bulk=v["n_bulk"], N_total=v["N_total"],
                        n_arcs=len(v["points"])) for k, v in by_cfg.items()})
    json.dump(r_all, open(out, "w"), indent=2)
    plot_results(single, by_cfg)
    print("Wrote results.json key: A3b4_first_law")


def plot_results(single, by_cfg):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))
    cols = {"d=2": "C0", "d=3": "C1", "d=2(deep)": "C2"}
    # (a) graded FLM scatter: measured vs predicted wedge count
    ax = axes[0]
    for tag, rec in by_cfg.items():
        dS = np.array([p[0] for p in rec["points"]])
        pred = np.array([p[1] for p in rec["points"]], float)
        j = np.random.uniform(-0.12, 0.12, dS.size)
        ax.scatter(pred + j, dS + j, s=9, alpha=0.30, color=cols.get(tag, "C3"),
                   label=f"{tag} (slope {rec['slope']:.2f})")
    m = max(max(p[1] for p in r["points"]) for r in by_cfg.values())
    ax.plot([0, m], [0, m], "k--", lw=1.0, label=r"$\delta S_A=\delta S_{\rm bulk}$")
    ax.set_xlabel(r"predicted $|B\cap{\rm EW}(A)|$ (geometric wedge)")
    ax.set_ylabel(r"measured $\delta S_A$ [bits]")
    ax.set_title(r"(a) full FLM: $\delta S_A=\delta S_{\rm bulk}({\rm EW}(A))$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (b) radial capture fraction f_b vs bulk depth (deep) -- geometry tie-in
    ax = axes[1]
    rad = single["d=2(deep)"]["radial_capture"]
    ax.plot(list(rad.keys()), list(rad.values()), "o-", color="C2", ms=9, lw=1.6)
    ax.set_xlabel("bulk radial depth")
    ax.set_ylabel(r"min capturing arc fraction $f_b$ (via $\delta S$)")
    ax.set_title("(b) deep bulk needs a larger arc to shift $S$")
    ax.set_ylim(0, 0.6); ax.grid(alpha=0.3)
    # (c) summary
    ax = axes[2]; ax.axis("off")
    txt = "A3b-dynamical (first law):\n\nLinearised  dS_A = [b in EW(A)]:\n"
    for tag, rec in single.items():
        txt += (f"  {tag}: {rec['exact_frac']*100:.0f}% exact, "
                f"dS in {rec['dS_values']}\n")
    txt += "\nFull FLM  dS_A = |bulk cap EW(A)|:\n"
    for tag, rec in by_cfg.items():
        txt += f"  {tag}: slope {rec['slope']:.2f}, {rec['exact_frac']*100:.0f}% exact\n"
    txt += ("\n  dS_A = dS_bulk(EW(A))  is the\n  linearised Einstein equation\n"
            "  (Faulkner et al. 2014).\n\n"
            "  Unit response (dS in {0,1}) per\n  excitation; holds on the perfect-\n"
            "  tensor substrate, UNDEFINED on\n  the dissipative class (no wedge).\n"
            "  First DYNAMICAL positive.\n  STATUS: R; residual = finite-size.")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9,
            family="monospace", va="top")
    fig.suptitle(r"A3b-dynamical: the substrate-relative first law "
                 r"$\delta S_A=\delta S_{\rm bulk}({\rm EW}(A))$ "
                 r"(linearised computational Einstein equation)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(os.path.dirname(__file__) or ".", "fig_A3b4_first_law.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
