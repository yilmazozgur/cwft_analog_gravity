"""
A2 phase 2 -- the Clausius/Einstein test with power-law non-locality.

Two parallel runs on the SAME 61x61 reservoir lattice, SAME family of
horizons (4 widths x 6 depths = 24 black holes), SAME gain-field estimator
(local-Jacobian spectral radius, as in cwf_clausius2d.py and Test 6):

  (i)  baseline  : chaotic NN coupling (alpha = infinity)
  (ii) non-local : power-law coupling K(r) ~ |r|^-3
                   (the fast-scrambling sweet spot from phase 1)

Both at the *chaotic* baseline (c=0.50, rho=2.6, m=10). NOTE: this differs
from the original cwf_clausius2d.py which used c=0.05, a regime we
discovered (phase 1) is globally CONTRACTIVE (lambda ~ -0.2) despite
having positive local-Jacobian gain. Phase 1's chaotic regime is the
proper baseline against which to ask "does adding fast scrambling change
the area-law failure?"

For each run we report:
  - mass-area scaling exponent  p:  M_c ~ A^p
  - per-width-slice eta_c       (universal => Clausius holds)
  - overall eta_c CV            (low CV => universal)
  - first-law fit R^2

Resurrection signature:
  - eta_c becomes universal across width-slices (low CV, high R^2);
  - p drops from ~3 toward ~1/2 (the GR Schwarzschild value).

Deepened-negative signature:
  - eta_c stays non-universal; p stays large.
  => non-locality is not sufficient; the path to Einstein dynamics also
     needs reversibility (=> A3, Clifford lattice horizon).
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ substrate
# Match cwf_a2_phase1_scrambling.py exactly (chaotic baseline).
M, RHO, C = 10, 2.6, 0.50
_RNG_INIT = np.random.default_rng(0)
_W = _RNG_INIT.standard_normal((M, M)); _W /= max(abs(np.linalg.eigvals(_W)))
_P = _RNG_INIT.standard_normal((M, M)); _P /= max(abs(np.linalg.eigvals(_P)))
_EYE = np.eye(M)

N = 61
cx = cy = N // 2
Y, X = np.mgrid[0:N, 0:N]
RAD = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

def well(Lmax, w):
    return Lmax * np.exp(-(RAD / w) ** 2)

# ---- two coupling architectures -------------------------------------------
def step_nn(H, leak):
    """alpha = inf: nearest-neighbour coupling (4 neighbours)."""
    nb = (np.roll(H, 1, 0) + np.roll(H, -1, 0)
          + np.roll(H, 1, 1) + np.roll(H, -1, 1))
    pre = RHO * (H @ _W.T) + C * (nb @ _P.T) - leak[..., None] * H
    return np.tanh(pre), pre

def make_kernel(N_, alpha):
    """Translation-invariant power-law kernel, sum-normalized to 4."""
    idx = np.arange(N_)
    idx = np.where(idx <= N_ // 2, idx, idx - N_)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    r = np.sqrt(I.astype(float) ** 2 + J.astype(float) ** 2)
    K = np.zeros_like(r)
    nz = r > 0
    K[nz] = 1.0 / (r[nz] ** alpha)
    K *= 4.0 / K.sum()
    return K

_K_ALPHA3 = make_kernel(N, 3.0)
_K_ALPHA3_FFT = np.fft.rfft2(_K_ALPHA3)

def step_pl3(H, leak):
    """alpha = 3 power-law (fast-scrambling regime from phase 1)."""
    Hf = np.fft.rfft2(H, axes=(0, 1))
    nb = np.fft.irfft2(Hf * _K_ALPHA3_FFT[..., None], s=(N, N), axes=(0, 1))
    pre = RHO * (H @ _W.T) + C * (nb @ _P.T) - leak[..., None] * H
    return np.tanh(pre), pre

# ---- shared gain-field estimator (same as cwf_clausius2d.py) ---------------
def gain_field(leak, step_fn, T=240, burn=64, sample=8, seed=0):
    """Time-averaged log spectral radius of the LOCAL Jacobian block.

    Same estimator as cwf_clausius2d.py and Test 6. Measures local block
    expansion (sech^2 * (rho*W - leak*I)), NOT the full coupled Lyapunov.
    This is the gain field whose g=0 surface is the computational horizon.
    """
    rng = np.random.default_rng(seed)
    H = 0.1 * rng.standard_normal((N, N, M))
    acc = np.zeros((N, N)); cnt = 0
    Mbase = RHO * _W[None, None] - leak[..., None, None] * _EYE[None, None]
    for t in range(T):
        H, pre = step_fn(H, leak)
        if t >= burn and (t - burn) % sample == 0:
            sech2 = 1.0 - np.tanh(pre) ** 2
            B = sech2[..., :, None] * Mbase
            ev = np.linalg.eigvals(B.reshape(-1, M, M))
            sr = np.abs(ev).max(axis=1).reshape(N, N)
            acc += np.log(sr + 1e-30); cnt += 1
    return acc / max(cnt, 1)

def radial_avg(field, nbins=34):
    bins = np.linspace(0, RAD.max(), nbins + 1)
    idx = np.digitize(RAD.ravel(), bins) - 1
    f = field.ravel()
    g = np.array([f[idx == k].mean() if np.any(idx == k) else np.nan
                  for k in range(nbins)])
    rc = 0.5 * (bins[:-1] + bins[1:])
    ok = ~np.isnan(g)
    return rc[ok], g[ok]

def horizon_props(gf):
    rc, g = radial_avg(gf)
    gs = np.convolve(g, np.ones(3) / 3, mode="same")
    cr = np.where((gs[:-1] < 0) & (gs[1:] >= 0))[0]
    if cr.size == 0 or gs[0] >= 0:
        return None
    k = cr[0]
    r_h = rc[k] + (rc[k + 1] - rc[k]) * (0 - gs[k]) / (gs[k + 1] - gs[k])
    lo, hi = max(0, k - 3), min(len(rc), k + 5)
    slope = np.polyfit(rc[lo:hi], gs[lo:hi], 1)[0]
    kappa = 0.5 * abs(slope)
    A = 2 * np.pi * r_h
    Tc = kappa / (2 * np.pi)
    Mc = float(np.sum(np.clip(-gf, 0, None)[RAD < r_h]))
    return dict(r_h=float(r_h), A=float(A), kappa=float(kappa),
                Tc=float(Tc), Mc=Mc)

# ---- family measurement ----------------------------------------------------
def measure_family(label, step_fn, widths, depths, seeds=3, verbose=True):
    sweep = []
    t0 = time.time()
    for w in widths:
        for Lmax in depths:
            leak = well(Lmax, w)
            gf = np.mean([gain_field(leak, step_fn, seed=10 + s)
                          for s in range(seeds)], axis=0)
            props = horizon_props(gf)
            if props is None:
                if verbose:
                    print(f"  [{label}] w={w:4.1f} Lmax={Lmax:.1f}  NO HORIZON")
                continue
            props.update(Lmax=float(Lmax), w=float(w), label=label)
            sweep.append(props)
            if verbose:
                print(f"  [{label}] w={w:4.1f} Lmax={Lmax:.1f}  "
                      f"r_h={props['r_h']:6.2f}  A={props['A']:7.1f}  "
                      f"T_c={props['Tc']:.4f}  M_c={props['Mc']:8.1f}")
    if verbose:
        print(f"  ({len(sweep)} horizons, {time.time()-t0:.0f}s)\n")
    return sweep

# ---- Clausius analysis -----------------------------------------------------
def analyze(sweep, rh_min=5.0):
    if not sweep:
        return dict(n=0)
    A = np.array([s["A"] for s in sweep])
    M_ = np.array([s["Mc"] for s in sweep])
    p_fit = float(np.polyfit(np.log(A), np.log(M_), 1)[0])
    by_w = {}
    for s in sweep:
        by_w.setdefault(s["w"], []).append(s)
    eta_per_w = {}
    for w, sl in by_w.items():
        sl = sorted([s for s in sl if s["r_h"] >= rh_min],
                    key=lambda d: d["A"])
        es = [(b["Mc"] - a["Mc"]) /
              (0.5 * (a["Tc"] + b["Tc"]) * (b["A"] - a["A"]))
              for a, b in zip(sl[:-1], sl[1:]) if abs(b["A"] - a["A"]) > 1e-3]
        if es:
            eta_per_w[float(w)] = (float(np.median(es)), len(es))
    # cross-width Clausius fit
    all_etas, pts = [], []
    for w, sl in by_w.items():
        sl = sorted([s for s in sl if s["r_h"] >= rh_min],
                    key=lambda d: d["A"])
        for a, b in zip(sl[:-1], sl[1:]):
            dA = b["A"] - a["A"]; dM = b["Mc"] - a["Mc"]
            Tc = 0.5 * (a["Tc"] + b["Tc"])
            if abs(dA) > 1e-3 and Tc > 0:
                all_etas.append(dM / (Tc * dA))
                pts.append((Tc * dA, dM))
    e = np.array(all_etas); pts = np.array(pts)
    if e.size >= 2:
        cv = float(np.std(e) / (abs(np.mean(e)) + 1e-12))
        x, y = pts[:, 0], pts[:, 1]
        slope = float(np.sum(x * y) / np.sum(x * x))
        r2 = float(1 - np.sum((y - slope * x) ** 2)
                   / np.sum((y - y.mean()) ** 2))
        eta_median = float(np.median(e))
    else:
        cv = float("nan"); slope = float("nan"); r2 = float("nan")
        eta_median = float("nan")
    return dict(n=len(sweep), mass_area_p=p_fit,
                eta_per_w=eta_per_w, eta_cv=cv, eta_median=eta_median,
                clausius_slope=slope, clausius_R2=r2,
                A_arr=A.tolist(), M_arr=M_.tolist())

def verdict_from(res):
    cv = res.get("eta_cv", float("nan"))
    r2 = res.get("clausius_R2", float("nan"))
    if not np.isfinite(cv):
        return "INSUFFICIENT"
    if cv < 0.25 and r2 > 0.85:
        return "CLAUSIUS HOLDS (near-universal eta_c)"
    if cv < 0.5:
        return "PARTIAL: approximate first law"
    return "CLAUSIUS FAILS"

# ---- main driver -----------------------------------------------------------
def main():
    # At c=0.50 the coupling drive (~2.0) is comparable to the recurrence
    # drive (~2.6), so the leak needs to be substantially larger than at
    # c=0.05 to saturate sech^2 and pull g(x) below zero (a quick probe at
    # the original depths [3.9..7.4] found ZERO horizons). Widened depth
    # range; same width sweep.
    widths = [8.0, 11.0, 14.0, 17.0]
    depths = [8.0, 12.0, 18.0, 25.0, 35.0, 50.0]
    print(f"A2 phase 2  --  Clausius test, chaotic baseline (c={C}, rho={RHO})\n")

    print("(1) BASELINE: nearest-neighbour (alpha=inf)")
    sweep_nn = measure_family("NN", step_nn, widths, depths, seeds=2)
    res_nn = analyze(sweep_nn)
    print("(2) NON-LOCAL: power-law alpha=3 (fast-scrambling regime from phase 1)")
    sweep_pl = measure_family("alpha=3", step_pl3, widths, depths, seeds=2)
    res_pl = analyze(sweep_pl)

    v_nn = verdict_from(res_nn); v_pl = verdict_from(res_pl)
    def fmt_w(eta_per_w):
        if not eta_per_w: return "(none)"
        return " | ".join(f"w={w:.0f}:{e[0]:7.1f}" for w, (e_, n) in
                         sorted(eta_per_w.items()) for e in [(e_, n)])
    def fmt_res(label, res):
        if res.get("n", 0) == 0:
            return f"  {label}: 0 horizons formed  -> NO_HORIZON_REGIME"
        p = res.get("mass_area_p", float("nan"))
        em = res.get("eta_median", float("nan"))
        cv = res.get("eta_cv", float("nan"))
        r2 = res.get("clausius_R2", float("nan"))
        return (f"  {label} (n={res['n']:2d}): "
                f"M_c ~ A^{p:.2f}, eta median={em:.1f}, CV={cv:.2f}, "
                f"R^2={r2:.2f}")
    print("=" * 78)
    print(fmt_res(f"baseline NN", res_nn))
    if res_nn.get("eta_per_w"):
        print(f"    per-width eta_c: {fmt_w(res_nn['eta_per_w'])}")
    print(f"    -> {v_nn}")
    print(fmt_res(f"alpha=3     ", res_pl))
    if res_pl.get("eta_per_w"):
        print(f"    per-width eta_c: {fmt_w(res_pl['eta_per_w'])}")
    print(f"    -> {v_pl}")
    print("=" * 78)

    # compare to original Clausius2D (c=0.05, contractive)
    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    orig = r_all.get("ClausiusTest2D", {})
    if orig:
        print("  reference (original c=0.05 contractive substrate):")
        print(f"    M_c ~ A^{orig['mass_area_exponent_p']:.2f}, "
              f"eta median={orig['eta_median']:.1f}, CV={orig['eta_cv']:.2f}, "
              f"R^2={orig['clausius_fit_R2']:.2f}  -> {orig['verdict']}")

    # --- summarize movement of the figures of merit -------------------------
    def diff(orig_v, new_v):
        if not np.isfinite(orig_v) or not np.isfinite(new_v): return ""
        return f"({orig_v:+.2f} -> {new_v:+.2f})"
    if orig:
        print("\n  movement vs original (c=0.05 contractive):")
        if res_nn.get("n", 0) > 0:
            print(f"    mass-area p     {diff(orig['mass_area_exponent_p'], res_nn['mass_area_p'])}  [NN c=0.50]")
            print(f"    eta_c CV        {diff(orig['eta_cv'], res_nn['eta_cv'])}      [NN c=0.50]")
        if res_pl.get("n", 0) > 0:
            print(f"    mass-area p     {diff(orig['mass_area_exponent_p'], res_pl['mass_area_p'])}  [alpha=3 c=0.50]")
            print(f"    eta_c CV        {diff(orig['eta_cv'], res_pl['eta_cv'])}      [alpha=3 c=0.50]")

    r_all["A2_phase2_clausius_nonlocal"] = dict(
        grid=N, m=M, rho=RHO, coupling=C,
        baseline_NN=res_nn, nonlocal_alpha3=res_pl,
        verdict_NN=v_nn, verdict_alpha3=v_pl,
        sweep_NN=sweep_nn, sweep_alpha3=sweep_pl,
    )
    json.dump(r_all, open(out, "w"), indent=2)

    # --- figures -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    colors_w = {8: "C0", 11: "C1", 14: "C2", 17: "C3"}
    for ax, sw, res, tag in [(axes[0], sweep_nn, res_nn, f"NN (c={C})"),
                              (axes[1], sweep_pl, res_pl, f"alpha=3 (c={C})")]:
        if not sw:
            ax.text(0.5, 0.5, "(no horizons formed)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=14)
            ax.set_title(f"{tag}: NO HORIZONS")
            ax.set_xlabel("horizon area $A = 2\\pi r_h$")
            ax.set_ylabel("computational mass $M_c$")
            continue
        by_w = {}
        for s in sw: by_w.setdefault(s["w"], []).append(s)
        for w, sl in sorted(by_w.items()):
            sl = sorted(sl, key=lambda d: d["A"])
            ax.plot([s["A"] for s in sl], [s["Mc"] for s in sl], "o-",
                    color=colors_w.get(int(w), "k"), label=f"w={int(w)}", ms=5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("horizon area $A = 2\\pi r_h$")
        ax.set_ylabel("computational mass $M_c$")
        p = res.get("mass_area_p", float("nan"))
        cv = res.get("eta_cv", float("nan"))
        r2 = res.get("clausius_R2", float("nan"))
        ax.set_title(f"{tag}: $M_c \\sim A^{{{p:.2f}}}$\n"
                     f"$\\eta_c$ CV={cv:.2f}, $R^2$={r2:.2f}")
        ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    plt.suptitle("A2 phase 2: does fast scrambling (alpha=3) resurrect Clausius?",
                 fontsize=11)
    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__) or ".",
                            "fig_A2_phase2_clausius.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\nWrote {fig_path}")
    print(f"Wrote results.json key: A2_phase2_clausius_nonlocal")


if __name__ == "__main__":
    main()
