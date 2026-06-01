"""
A1 -- reversibility / hysteresis test of computational-horizon thermodynamics.

Builds on the cwf_clausius2d.py infrastructure (same 61x61 reservoir lattice,
same gain-field estimator, same radial-average analysis), but instead of taking
independent snapshots at unrelated well configurations, we run a CLOSED
THERMODYNAMIC CYCLE in well-depth Lmax:

    Lmax:  L_low --(ramp up)-->  L_high  --(ramp down)-->  L_low

with the substrate state H carried CONTINUOUSLY through the entire cycle (no
burn-in reset between levels). At each Lmax level we hold leak fixed for
`steps_per_level` steps and measure the gain field over the final window of
that level. This yields an up-leg arc and a down-leg arc in the (A, M_c, T_c)
space.

Three independent hysteresis estimators:

  1. Loop area in (A, M_c) plane:   oint M_c dA       (signed)
  2. Clausius integral:             oint dM_c / T_c   (entropy produced)
  3. Pointwise gap at matched Lmax: |X_up(L) - X_dn(L)| for X in {A, M_c, T_c}

The ramp-rate sweep (varying steps_per_level) distinguishes:
  - equilibration lag  -> loop area --> 0 as ramp slows
  - genuine irreversibility -> loop area saturates at finite value

This is an independent second failure mode of the computational Clausius
relation (separate from the non-universal-eta_c / area-law failure already
established by cwf_clausius2d.py).
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ substrate
# Match cwf_clausius2d.py exactly so results are directly comparable.
RNG = np.random.default_rng(0)
N, M, RHO, C = 61, 10, 2.6, 0.05
_W = RNG.standard_normal((M, M)); _W /= max(abs(np.linalg.eigvals(_W)))
_P = RNG.standard_normal((M, M)); _P /= max(abs(np.linalg.eigvals(_P)))
_EYE = np.eye(M)
cx = cy = N // 2
Y, X = np.mgrid[0:N, 0:N]
RAD = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

def well(Lmax, w):
    return Lmax * np.exp(-(RAD / w) ** 2)

def step(H, leak):
    nb = (np.roll(H, 1, 0) + np.roll(H, -1, 0) +
          np.roll(H, 1, 1) + np.roll(H, -1, 1))
    pre = RHO * (H @ _W.T) + C * (nb @ _P.T) - leak[..., None] * H
    return np.tanh(pre), pre

def gain_from_window(sech2_samples, leak_samples):
    """Average log spectral radius of the local Jacobian block over a window."""
    K = len(sech2_samples)
    acc = np.zeros((N, N))
    for s in range(K):
        sech2 = sech2_samples[s]; leak = leak_samples[s]
        Mbase = RHO * _W[None, None] - leak[..., None, None] * _EYE[None, None]
        B = sech2[..., :, None] * Mbase
        ev = np.linalg.eigvals(B.reshape(-1, M, M))
        sr = np.abs(ev).max(axis=1).reshape(N, N)
        acc += np.log(sr + 1e-30)
    return acc / max(K, 1)

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
    """Extract (r_h, A, kappa, T_c, M_c) from a gain field; None if no horizon."""
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

# ------------------------------------------------------------------ the cycle

def hysteresis_cycle(L_low, L_high, w, n_levels_per_leg=30,
                     steps_per_level=120, T_win=36, sample_stride=6,
                     warmup=200, seed=0, verbose=False):
    """
    One closed thermodynamic cycle: Lmax: L_low -> L_high -> L_low.

    State H is carried continuously through the whole cycle. At each level we
    hold leak fixed for `steps_per_level` steps; the final T_win steps of the
    level are sampled (every `sample_stride`) for the gain-field average. Then
    Lmax advances to the next level WITHOUT touching H.
    """
    levels = np.concatenate([np.linspace(L_low, L_high, n_levels_per_leg),
                             np.linspace(L_high, L_low, n_levels_per_leg)])
    legs = ['up'] * n_levels_per_leg + ['dn'] * n_levels_per_leg

    rng = np.random.default_rng(seed)
    H = 0.1 * rng.standard_normal((N, N, M))
    # warm up at L_low so the cycle starts from a near-stationary state
    leak0 = well(L_low, w)
    for _ in range(warmup):
        H, _ = step(H, leak0)

    snapshots = []
    meas_start = steps_per_level - T_win
    for k, (Lmax_k, leg_k) in enumerate(zip(levels, legs)):
        leak = well(Lmax_k, w)
        sech2_buf, leak_buf = [], []
        for s in range(steps_per_level):
            H, pre = step(H, leak)
            if s >= meas_start and (s - meas_start) % sample_stride == 0:
                sech2_buf.append(1.0 - np.tanh(pre) ** 2)
                leak_buf.append(leak)
        gf = gain_from_window(sech2_buf, leak_buf)
        props = horizon_props(gf)
        snap = dict(k=k, Lmax=float(Lmax_k), leg=leg_k,
                    has_horizon=props is not None)
        if props is None:
            snap.update(r_h=0.0, A=0.0, kappa=0.0, Tc=0.0, Mc=0.0)
        else:
            snap.update(props)
        snapshots.append(snap)
        if verbose:
            tag = "*" if props is not None else "-"
            print(f"    [{leg_k}{tag}] k={k:3d}  Lmax={Lmax_k:.3f}  "
                  f"r_h={snap['r_h']:5.2f}  A={snap['A']:6.1f}  "
                  f"T_c={snap['Tc']:.4f}  M_c={snap['Mc']:7.1f}")

    return snapshots

# ------------------------------------------------------------------ analysis

def loop_area_AM(snaps):
    """Signed shoelace area in (A, M_c). Up-leg + reversed down-leg = closed loop."""
    A = np.array([s["A"] for s in snaps])
    M_ = np.array([s["Mc"] for s in snaps])
    x = np.concatenate([A, [A[0]]])
    y = np.concatenate([M_, [M_[0]]])
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

def clausius_integral(snaps):
    """oint dM_c / T_c around the cycle. Skips steps where horizon absent."""
    dMc = 0.0
    for a, b in zip(snaps[:-1], snaps[1:]):
        if not (a["has_horizon"] and b["has_horizon"]):
            continue
        Tc_avg = 0.5 * (a["Tc"] + b["Tc"])
        if Tc_avg <= 0:
            continue
        dMc += (b["Mc"] - a["Mc"]) / Tc_avg
    # close the loop
    a, b = snaps[-1], snaps[0]
    if a["has_horizon"] and b["has_horizon"]:
        Tc_avg = 0.5 * (a["Tc"] + b["Tc"])
        if Tc_avg > 0:
            dMc += (b["Mc"] - a["Mc"]) / Tc_avg
    return dMc

def pointwise_gap(snaps, n_per_leg):
    """At matched Lmax (up-leg level k vs down-leg level 2*n - 1 - k), compute |dX|."""
    rows = []
    for k in range(n_per_leg):
        up = snaps[k]
        dn = snaps[2 * n_per_leg - 1 - k]
        rows.append(dict(Lmax=up["Lmax"],
                         dA=dn["A"] - up["A"],
                         dMc=dn["Mc"] - up["Mc"],
                         dTc=dn["Tc"] - up["Tc"],
                         both_have_horizon=(up["has_horizon"] and dn["has_horizon"])))
    return rows

# ------------------------------------------------------------------ driver

def run_sweep(out_path="results.json",
              fig_path="fig_A1_hysteresis.png",
              raw_path="a1_hysteresis_raw.json",
              L_low=0.0, L_high=6.0, w=14.0,
              n_levels_per_leg=30,
              ramp_speeds=(60, 120, 240, 480, 960),  # steps_per_level
              seeds=(0, 1, 2, 3),
              T_win=36, sample_stride=6,
              pilot=False):
    """
    pilot=True -> single ramp speed, 2 seeds, reduced n_levels -- for timing.
    """
    if pilot:
        ramp_speeds = (ramp_speeds[1],)
        seeds = seeds[:2]
        n_levels_per_leg = 15

    summary = []
    raw = {}
    for spl in ramp_speeds:
        T_cyc = 2 * n_levels_per_leg * spl
        per_seed = []
        loop_areas = []
        clausius = []
        t_start = time.time()
        for sd in seeds:
            snaps = hysteresis_cycle(
                L_low=L_low, L_high=L_high, w=w,
                n_levels_per_leg=n_levels_per_leg,
                steps_per_level=spl,
                T_win=min(T_win, spl // 2),
                sample_stride=sample_stride,
                seed=sd, verbose=False,
            )
            la = loop_area_AM(snaps)
            ci = clausius_integral(snaps)
            n_h = sum(1 for s in snaps if s["has_horizon"])
            loop_areas.append(la)
            clausius.append(ci)
            per_seed.append(dict(seed=sd, loop_area_AM=la,
                                 clausius_integral=ci, n_horizon=n_h,
                                 snapshots=snaps))
        dt = time.time() - t_start
        agg = dict(
            steps_per_level=spl,
            T_cyc=T_cyc,
            n_levels_per_leg=n_levels_per_leg,
            T_win=min(T_win, spl // 2),
            sample_stride=sample_stride,
            loop_area_AM_mean=float(np.mean(loop_areas)),
            loop_area_AM_std=float(np.std(loop_areas)),
            clausius_integral_mean=float(np.mean(clausius)),
            clausius_integral_std=float(np.std(clausius)),
            elapsed_s=dt,
        )
        summary.append(agg)
        raw[f"spl_{spl}"] = per_seed
        print(f"  spl={spl:4d}  T_cyc={T_cyc:6d}  "
              f"loop_area={agg['loop_area_AM_mean']:+10.1f} +/- {agg['loop_area_AM_std']:7.1f}  "
              f"clausius_int={agg['clausius_integral_mean']:+8.2f} +/- {agg['clausius_integral_std']:6.2f}  "
              f"({dt:.0f}s)")

    # --- decide outcome ---
    spls = np.array([a["steps_per_level"] for a in summary], float)
    LAs = np.array([abs(a["loop_area_AM_mean"]) for a in summary], float)
    if len(LAs) >= 2:
        # log-log slope of |loop area| vs steps_per_level
        slope = float(np.polyfit(np.log(spls), np.log(LAs + 1e-9), 1)[0])
    else:
        slope = float("nan")
    # ratio of slowest to fastest
    ratio = float(LAs[-1] / (LAs[0] + 1e-9)) if len(LAs) >= 2 else float("nan")
    if not np.isfinite(slope):
        verdict = "INCONCLUSIVE: insufficient ramp-rate range"
    elif ratio < 0.25:
        verdict = "EQUILIBRATION LAG: loop area shrinks with ramp slowdown (reversible substrate)"
    elif ratio > 0.7:
        verdict = "GENUINE IRREVERSIBILITY: loop area saturates with slow ramp"
    else:
        verdict = "PARTIAL: some equilibration lag but residual irreversibility"

    print(f"\n  loop-area log-log slope vs steps_per_level: {slope:+.2f}")
    print(f"  ratio |LA(slowest)| / |LA(fastest)|        : {ratio:.3f}")
    print(f"  VERDICT: {verdict}")

    # --- write results.json entry ---
    out_full = os.path.join(os.path.dirname(__file__) or ".", out_path)
    res = json.load(open(out_full)) if os.path.exists(out_full) else {}
    res["A1_hysteresis"] = dict(
        grid=N, m=M, rho=RHO, coupling=C,
        L_low=L_low, L_high=L_high, w=w,
        n_levels_per_leg=n_levels_per_leg,
        seeds=list(seeds),
        ramp_speeds=list(int(s) for s in spls),
        loop_area_AM_mean=[a["loop_area_AM_mean"] for a in summary],
        loop_area_AM_std=[a["loop_area_AM_std"] for a in summary],
        clausius_integral_mean=[a["clausius_integral_mean"] for a in summary],
        clausius_integral_std=[a["clausius_integral_std"] for a in summary],
        loop_area_loglog_slope=slope,
        loop_area_ratio_slow_over_fast=ratio,
        verdict=verdict,
    )
    json.dump(res, open(out_full, "w"), indent=2)

    # --- write raw snapshots ---
    raw_full = os.path.join(os.path.dirname(__file__) or ".", raw_path)
    json.dump(dict(summary=summary, per_seed_per_speed=raw),
              open(raw_full, "w"), indent=2)

    # --- figures ---
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

    # (1) loops in (A, M_c) for each ramp speed, seed 0
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(summary)))
    for spl, col in zip([a["steps_per_level"] for a in summary], colors):
        snaps = raw[f"spl_{spl}"][0]["snapshots"]
        up = [s for s in snaps if s["leg"] == "up"]
        dn = [s for s in snaps if s["leg"] == "dn"]
        ax.plot([s["A"] for s in up], [s["Mc"] for s in up],
                "o-", color=col, ms=3, label=f"spl={spl}  up", alpha=0.85)
        ax.plot([s["A"] for s in dn], [s["Mc"] for s in dn],
                "s--", color=col, ms=3, label=f"spl={spl}  dn", alpha=0.85)
    ax.set_xlabel("horizon area $A=2\\pi r_h$")
    ax.set_ylabel("computational mass $M_c$")
    ax.set_title("Cycle loops in $(A, M_c)$ -- seed 0")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    # (2) |loop area| vs ramp speed
    ax = axes[1]
    ax.errorbar([a["steps_per_level"] for a in summary],
                [abs(a["loop_area_AM_mean"]) for a in summary],
                yerr=[a["loop_area_AM_std"] for a in summary],
                fmt="o-", color="C3", capsize=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("steps per level (ramp slowness)")
    ax.set_ylabel("$|\\oint M_c\\,dA|$  (loop area)")
    ax.set_title(f"Loop area vs ramp speed\nlog-log slope $={slope:+.2f}$, ratio $={ratio:.2f}$")
    ax.grid(alpha=0.3, which="both")

    # (3) Clausius integral vs ramp speed
    ax = axes[2]
    means = [a["clausius_integral_mean"] for a in summary]
    stds = [a["clausius_integral_std"] for a in summary]
    ax.errorbar([a["steps_per_level"] for a in summary], means, yerr=stds,
                fmt="s-", color="C2", capsize=4)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("steps per level (ramp slowness)")
    ax.set_ylabel("$\\oint dM_c/T_c$  (Clausius integral)")
    ax.set_title("Entropy-production proxy")
    ax.grid(alpha=0.3)

    fig.suptitle(f"A1: hysteresis test  ($N$={N}, $w$={w}, $L\\in[{L_low},{L_high}]$, "
                 f"{n_levels_per_leg} levels/leg, {len(seeds)} seeds) -- {verdict}",
                 fontsize=10)
    plt.tight_layout()
    fig_full = os.path.join(os.path.dirname(__file__) or ".", fig_path)
    plt.savefig(fig_full, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\nWrote {fig_full}")
    print(f"Wrote {out_full}  (key: A1_hysteresis)")
    print(f"Wrote {raw_full}")

# ------------------------------------------------------------------ main

if __name__ == "__main__":
    import sys
    pilot = ("--pilot" in sys.argv)
    print("A1 -- computational-horizon hysteresis test"
          f"  ({'PILOT' if pilot else 'FULL'}) {N}x{N}, m={M}")
    run_sweep(pilot=pilot)
