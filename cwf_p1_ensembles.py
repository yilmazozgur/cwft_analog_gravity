#!/usr/bin/env python3
"""
P1-ensemble pass for the gravity paper (JPC revision round).

Adds the uncertainty / robustness / failure-classification layer the referee
simulations asked for, WITHOUT touching the committed results.json:

  A. Clausius 2D seed ensemble  -> eta_c medians, CV, exponent, R2: mean+-sd
                                   across gain-field seed bases; the decisive
                                   comparison: per-width eta spread vs seed sd.
  B. Hysteresis seed ensemble   -> loop area mean+-sd at the fastest and the
                                   slowest committed ramps + normalised loop.
  C. Test6 kinematics sweep     -> horizon/clock-freeze offset across a
                                   (depth, width, coupling) grid + seed spread.
  D. MIPT statistics            -> per-(N,p) I3 SEM at 200 realisations
                                   (same seeding as the committed sweep) +
                                   pairwise crossing drift from the committed
                                   record.
  E. A3b2 RT-arc diagnostics    -> one-sidedness (S <= |gamma|?) and where the
                                   non-exact arcs live.
  F. A3b4 first-law failures    -> classification of failing (excitation, arc)
                                   pairs: over/under response, excitation
                                   depth, adjacency to the wedge boundary.

Everything is seeded; output -> p1_ensemble_results.json (a NEW record).
"""
import json, time, importlib, numpy as np

OUT = {}
T0 = time.time()

def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ---------------------------------------------------------------- A. Clausius
log("A. Clausius 2D seed ensemble ...")
src = open("cwf_clausius2d.py").read()
head = src.split("t0 = time.time()")[0]
CL = {}
exec(head, CL)
widths = [8.0, 11.0, 14.0, 17.0]
depths = [3.9, 4.5, 5.2, 5.9, 6.6, 7.4]

def clausius_sweep(base):
    """Full committed sweep with gain-field seed base `base` (committed: 10)."""
    sweep = []
    for w in widths:
        for Lmax in depths:
            leak = CL['well'](Lmax, w)
            gf = np.mean([CL['gain_field'](leak, seed=base + s) for s in range(2)], axis=0)
            rc, g = CL['radial_avg'](gf)
            gs = np.convolve(g, np.ones(3) / 3, mode="same")
            cr = np.where((gs[:-1] < 0) & (gs[1:] >= 0))[0]
            if cr.size == 0 or gs[0] >= 0:
                continue
            k = cr[0]
            r_h = rc[k] + (rc[k + 1] - rc[k]) * (0 - gs[k]) / (gs[k + 1] - gs[k])
            lo, hi = max(0, k - 3), min(len(rc), k + 5)
            slope = np.polyfit(rc[lo:hi], gs[lo:hi], 1)[0]
            kappa = 0.5 * abs(slope)
            A = 2 * np.pi * r_h
            Mc = float(np.sum(np.clip(-gf, 0, None)[CL['RAD'] < r_h]))
            sweep.append(dict(w=w, Lmax=Lmax, r_h=float(r_h), A=float(A),
                              Tc=float(kappa / (2 * np.pi)), Mc=Mc))
    return sweep

def clausius_stats(sweep):
    allA = np.array([m["A"] for m in sweep]); allM = np.array([m["Mc"] for m in sweep])
    p_fit = float(np.polyfit(np.log(allA), np.log(allM), 1)[0])
    etas, pts, per_w = [], [], {}
    for w in widths:
        sl = sorted([m for m in sweep if m["w"] == w and m["r_h"] >= 5.0],
                    key=lambda d: d["A"])
        es = []
        for a, b in zip(sl[:-1], sl[1:]):
            dA = b["A"] - a["A"]; dM = b["Mc"] - a["Mc"]; Tc = 0.5 * (a["Tc"] + b["Tc"])
            if abs(dA) > 1e-3:
                etas.append(dM / (Tc * dA)); pts.append((Tc * dA, dM)); es.append(etas[-1])
        per_w[w] = float(np.median(es)) if es else None
    e = np.array(etas); pts = np.array(pts)
    cv = float(np.std(e) / (abs(np.mean(e)) + 1e-12))
    x, y = pts[:, 0], pts[:, 1]
    s = float(np.sum(x * y) / np.sum(x * x))
    r2 = float(1 - np.sum((y - s * x) ** 2) / np.sum((y - y.mean()) ** 2))
    return dict(exponent=p_fit, eta_median=float(np.median(e)), eta_cv=cv,
                fit_R2=r2, per_width_eta=per_w)

SEED_BASES = [10, 1010, 2010, 3010, 4010, 5010, 6010, 7010, 8010, 9010]  # 10 incl. committed
runs = []
for b in SEED_BASES:
    st = clausius_stats(clausius_sweep(b))
    st["seed_base"] = b
    runs.append(st)
    log(f"   base={b}: exp={st['exponent']:.2f} etaCV={st['eta_cv']:.2f} "
        f"R2={st['fit_R2']:.2f} per-w={ {k: round(v) for k, v in st['per_width_eta'].items()} }")

def msd(key):
    v = np.array([r[key] for r in runs]); return float(v.mean()), float(v.std())

per_w_mat = {w: np.array([r["per_width_eta"][w] for r in runs]) for w in widths}
width_spread = float(np.mean([per_w_mat[widths[-1]][i] / per_w_mat[widths[0]][i]
                              for i in range(len(runs))]))
seed_rel_sd = {str(w): float(per_w_mat[w].std() / per_w_mat[w].mean()) for w in widths}
OUT["clausius_ensemble"] = dict(
    n_seed_bases=len(SEED_BASES), seed_bases=SEED_BASES, runs=runs,
    exponent_mean_sd=msd("exponent"), eta_cv_mean_sd=msd("eta_cv"),
    fit_R2_mean_sd=msd("fit_R2"), eta_median_mean_sd=msd("eta_median"),
    per_width_eta_mean={str(w): float(per_w_mat[w].mean()) for w in widths},
    per_width_eta_sd={str(w): float(per_w_mat[w].std()) for w in widths},
    widest_to_narrowest_eta_ratio_mean=width_spread,
    per_width_seed_relative_sd=seed_rel_sd,
    note="per-width eta spread (the non-collapse) vs seed-to-seed sd is the "
         "decisive robustness comparison")
json.dump(OUT, open("p1_ensemble_results.json","w"), indent=2)
log(f"A done. exponent={msd('exponent')} eta_cv={msd('eta_cv')} "
    f"width-spread x{width_spread:.2f}, per-width seed rel-sd {seed_rel_sd}")

# -------------------------------------------------------------- B. Hysteresis
log("B. Hysteresis seed ensemble ...")
A1 = importlib.import_module("cwf_a1_hysteresis")
hyst = {}
for spl in (60, 960):
    loops, norms = [], []
    for sd in range(5):
        snaps = A1.hysteresis_cycle(L_low=0.0, L_high=6.0, w=14.0,
                                    steps_per_level=spl, seed=sd)
        la = abs(A1.loop_area_AM(snaps))
        Ms = np.array([s["Mc"] for s in snaps]); As = np.array([s["A"] for s in snaps])
        norm = la / ((Ms.max() - Ms.min()) * (As.max() - As.min()) + 1e-12)
        loops.append(float(la)); norms.append(float(norm))
        log(f"   spl={spl} seed={sd}: |loop|={la:.0f} normalised={norm:.3f}")
    hyst[str(spl)] = dict(loop_abs_mean=float(np.mean(loops)), loop_abs_sd=float(np.std(loops)),
                          loop_norm_mean=float(np.mean(norms)), loop_norm_sd=float(np.std(norms)),
                          n_seeds=5, loops=loops, norms=norms)
OUT["hysteresis_ensemble"] = hyst
json.dump(OUT, open("p1_ensemble_results.json","w"), indent=2)
log("B done.")

# ------------------------------------------------------------ C. Test6 sweep
log("C. Test6 kinematics parameter sweep ...")
t6src = open("cwf_test6b.py").read()
t6head = t6src.split("n, c, w, Lmax, rho = 301")[0]
T6 = {}
exec(t6head, T6)
THETA = 1e-4

def one_kinematics(n=301, c=150, w=18.0, Lmax=3.0, coupling=0.18, lat_seed=970):
    leak = T6['make_well'](n, c, w, Lmax)
    lat = T6['ReservoirLattice'](n, m=24, coupling=coupling, rho=2.6, seed=lat_seed)
    lat.leak = leak
    H0 = 0.1 * np.random.default_rng(lat_seed).standard_normal((n, 24))
    g = T6['jacobian_gain_profile'](lat, H0, T=400)
    reach, inj = T6['transport_reach'](n, leak, c, coupling=coupling)
    cross = np.where(np.diff(np.sign(g)) != 0)[0]
    g_h = int(cross[cross < c][-1]) if np.any(cross < c) else None
    reached = np.where(reach > 0.5)[0]
    t_h = int(reached[reached <= c].max()) if np.any(reached <= c) else None
    if g_h is None or t_h is None:
        return None
    return dict(w=w, Lmax=Lmax, coupling=coupling, seed=lat_seed,
                g_horizon=g_h, transport_horizon=t_h, offset=abs(g_h - t_h))

grid = []
for w in (12.0, 18.0, 24.0):
    for Lmax in (2.5, 3.0, 4.0):
        for cc in (0.14, 0.18, 0.22):
            r = one_kinematics(w=w, Lmax=Lmax, coupling=cc)
            if r:
                grid.append(r)
                log(f"   w={w} Lmax={Lmax} c={cc}: offset={r['offset']} "
                    f"(g@{r['g_horizon']}, t@{r['transport_horizon']})")
            else:
                log(f"   w={w} Lmax={Lmax} c={cc}: no horizon formed")
seed_runs = [one_kinematics(lat_seed=970 + 100 * k) for k in range(5)]
seed_offsets = [r["offset"] for r in seed_runs if r]
offs = [r["offset"] for r in grid]
OUT["test6_sweep"] = dict(
    grid=grid, n_configs=len(grid),
    offset_mean=float(np.mean(offs)), offset_max=int(np.max(offs)),
    offset_median=float(np.median(offs)),
    frac_within_5_sites=float(np.mean(np.array(offs) <= 5)),
    seed_offsets=seed_offsets,
    note="offset = |g(x)=0 crossing - transport-collapse site| across the "
         "(width, depth, coupling) grid at fixed centre c=150, n=301")
json.dump(OUT, open("p1_ensemble_results.json","w"), indent=2)
log(f"C done. offsets: median={np.median(offs)} max={np.max(offs)} "
    f"within5={np.mean(np.array(offs) <= 5):.2f} seeds={seed_offsets}")

# ---------------------------------------------------------------- D. MIPT
log("D. MIPT SEM + pairwise crossings ...")
PF = {"__name__": "not_main", "__file__": "cwf_ap_phaseF_twoknob.py"}
exec(open("cwf_ap_phaseF_twoknob.py").read(), PF)
N_list = [8, 12, 16, 20, 24]
p_cross = [0.11, 0.13, 0.15, 0.17, 0.19, 0.22]
sem = {}
for N in N_list:
    T = 3 * N
    row = {}
    for p in p_cross:
        rng = np.random.default_rng(0 * 100003 + N * 1009 + int(round(p * 1e6)))
        vals = [PF['run_realization'](N, p, T, rng)[1] for _ in range(200)]
        row[str(p)] = dict(mean=float(np.mean(vals)),
                           sem=float(np.std(vals) / np.sqrt(len(vals))),
                           sd=float(np.std(vals)), n=200)
    sem[str(N)] = row
    log(f"   N={N}: I3 SEM at p=0.16-region: " +
        " ".join(f"{p}:{row[str(p)]['sem']:.3f}" for p in p_cross))

rec = json.load(open("ap_phaseF_twoknob_results.json"))
I3 = rec["I3"]; p_grid = rec["setup"]["p_grid"]
def crossing(Na, Nb):
    a = np.array(I3[str(Na)]); b = np.array(I3[str(Nb)]); d = a - b
    for i in range(len(p_grid) - 1):
        if d[i] * d[i + 1] < 0 and p_grid[i] >= 0.05:
            return float(p_grid[i] + (p_grid[i + 1] - p_grid[i]) * d[i] / (d[i] - d[i + 1]))
    return None
pair_cross = {f"{a}-{b}": crossing(a, b)
              for i, a in enumerate(N_list) for b in N_list[i + 1:]}
vals = [v for v in pair_cross.values() if v]
OUT["mipt_stats"] = dict(
    sem_by_N=sem, reals=200,
    protocol=dict(circuit="brickwork random two-qubit Clifford, periodic",
                  measurement="projective Z at rate p per site per layer",
                  depth="T = 3N layers", partition="four contiguous quarters",
                  realisations=200, source="cwf_ap_phaseF_twoknob.py"),
    pairwise_crossings=pair_cross,
    crossing_mean=float(np.mean(vals)), crossing_sd=float(np.std(vals)),
    crossing_range=[float(np.min(vals)), float(np.max(vals))])
json.dump(OUT, open("p1_ensemble_results.json","w"), indent=2)
log(f"D done. pairwise crossings: {pair_cross} -> "
    f"{np.mean(vals):.3f} +- {np.std(vals):.3f}")

# ---------------------------------------------------------------- E. A3b2
log("E. A3b2 RT-arc diagnostics ...")
B2 = {"__name__": "not_main", "__file__": "cwf_a3b2_happy_perfect.py"}
exec(open("cwf_a3b2_happy_perfect.py").read(), B2)
diag = {}
for d in (2, 3, 4):
    r = B2['run_depth'](d, max_per_size=(None if d < 4 else 24))
    pts = r["points"]; nb = r["n_boundary"]
    dev = [(p["S"] - p["cut"], p["L"]) for p in pts]
    over = [x for x, _ in dev if x > 1e-9]           # S > cut would be a bug
    under = [(x, L) for x, L in dev if x < -1e-9]    # deficit side
    diag[str(d)] = dict(
        n_arcs=len(pts), n_boundary=nb,
        n_S_gt_cut=len(over), n_S_lt_cut=len(under),
        exact_frac=float(1 - len(under + [1 for _ in over]) / len(pts))
                   if pts else None,
        deficit_mean_bits=float(np.mean([-x for x, _ in under])) if under else 0.0,
        deficit_arc_fraction_mean=float(np.mean([L / nb for _, L in under])) if under else None,
        all_arc_fraction_mean=float(np.mean([p["L"] / nb for p in pts])))
    log(f"   depth={d}: arcs={len(pts)} S>cut:{len(over)} S<cut:{len(under)} "
        f"deficit-arc-frac={diag[str(d)]['deficit_arc_fraction_mean']}")
OUT["a3b2_diagnostics"] = dict(
    by_depth=diag,
    note="S>cut violations would contradict the min-cut bound (bug); "
         "S<cut deficits are the lawful direction. deficit_arc_fraction shows "
         "where the non-exact arcs live relative to boundary size.")
json.dump(OUT, open("p1_ensemble_results.json","w"), indent=2)
log("E done.")

# ---------------------------------------------------------------- F. A3b4
log("F. A3b4 first-law failure classification ...")
B4 = {"__name__": "not_main", "__file__": "cwf_a3b4_first_law.py"}
exec(open("cwf_a3b4_first_law.py").read(), B4)
BULK_LEG = B4['BULK_LEG']

def classify(max_depth, deep, cap):
    tensors, bonds, bulk_qubits, boundary = B4['build_network'](max_depth, 2, deep=deep)
    N = 6 * len(tensors); n = len(boundary)
    tid_of = {t.qubits[BULK_LEG]: t.tid for t in tensors}
    depth_of = {t.qubits[BULK_LEG]: t.depth for t in tensors}
    adj = {}
    for (ta, _), (tb, _) in bonds:
        adj.setdefault(ta, set()).add(tb); adj.setdefault(tb, set()).add(ta)
    simv, _ = B4['build_state'](tensors, bonds, bulk_qubits, fix_bulk=True)
    Xv, Zv = B4['stabilizer_matrix'](simv, N)
    arcs = list(B4['circular_arcs'](n, cap))
    wedges = [B4['wedge_tensors'](tensors, bonds, boundary,
                                  [boundary[(s + i) % n] for i in range(L)])
              for (L, s) in arcs]
    fail = dict(total=0, n_fail=0, under=0, over=0,
                fail_at_wedge_boundary=0, fail_by_depth={})
    for b in bulk_qubits:
        sime, _ = B4['build_state'](tensors, bonds, bulk_qubits, fix_bulk=True, excite=[b])
        Xe, Ze = B4['stabilizer_matrix'](sime, N)
        tb = tid_of[b]
        for (L, s), W in zip(arcs, wedges):
            A = [boundary[(s + i) % n] for i in range(L)]
            dS = B4['entropy_region'](Xe, Ze, A, N) - B4['entropy_region'](Xv, Zv, A, N)
            pred = 1 if tb in W else 0
            fail["total"] += 1
            if abs(dS - pred) < 1e-9:
                continue
            fail["n_fail"] += 1
            if pred == 1 and dS < 0.5:
                fail["under"] += 1
            elif pred == 0 and dS >= 0.5:
                fail["over"] += 1
            on_boundary = (tb in W and any(x not in W for x in adj.get(tb, ()))) or \
                          (tb not in W and any(x in W for x in adj.get(tb, ())))
            if on_boundary:
                fail["fail_at_wedge_boundary"] += 1
            dd = str(depth_of[b])
            fail["fail_by_depth"][dd] = fail["fail_by_depth"].get(dd, 0) + 1
    fail["exact_frac"] = 1 - fail["n_fail"] / fail["total"]
    fail["frac_failures_at_wedge_boundary"] = (fail["fail_at_wedge_boundary"] /
                                               fail["n_fail"]) if fail["n_fail"] else None
    return fail

flcls = {}
for tag, (d, deep, cap) in {"d2": (2, False, None), "d3": (3, False, 24),
                            "d2deep": (2, True, 16)}.items():
    flcls[tag] = classify(d, deep, cap)
    log(f"   {tag}: exact={flcls[tag]['exact_frac']:.4f} "
        f"fails={flcls[tag]['n_fail']} under={flcls[tag]['under']} "
        f"over={flcls[tag]['over']} at-wedge-boundary="
        f"{flcls[tag]['frac_failures_at_wedge_boundary']}")
OUT["a3b4_failure_classification"] = dict(
    by_config=flcls,
    note="under = wedge predicts response, none measured; over = response "
         "outside predicted wedge; wedge-boundary adjacency = the excited "
         "tensor sits on the wedge's edge (degenerate/ambiguous-cut region).")
json.dump(OUT, open("p1_ensemble_results.json","w"), indent=2)
log("F done.")

json.dump(OUT, open("p1_ensemble_results.json", "w"), indent=2)
log(f"WROTE p1_ensemble_results.json ({time.time()-T0:.0f}s total)")
