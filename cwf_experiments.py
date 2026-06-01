"""
CWF computational-black-hole experiment battery.

Tests (Section 7 agenda, reduced to CPU scale):
  E1  Butterfly velocity v_B as a function of recurrence inertia rho
      (homogeneous lattices). Question: does high recurrence SLOW
      information propagation (toward a horizon) or SPEED it (chaos spreads)?
  E2  Trapping-surface test: a lattice with a central high-rho band.
      Inject a perturbation at the band centre; does the divergence front
      escape the band, or is it trapped/slowed at the band boundary?
      Also a saturated-band variant (high leak) as an alternative mechanism.
  E3  Scrambling-time scaling t_* vs system size N. Fast-scrambler
      signature is t_* ~ log N.
  E4  Rule 110 exact perturbation light-cone (the substrate has a finite c).

Honest methodology notes:
  * Butterfly front = outermost site whose ref/perturbed divergence first
    exceeds a small threshold theta, with a tiny initial perturbation eps.
    We extract v_B from the linear regime of front-position vs time, before
    the front reaches the boundary.
  * Results are reported as measured; the hypothesis (high rho -> trapping)
    may be confirmed or refuted.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_substrate import ReservoirLattice, rule110_run

RNG = np.random.default_rng(7)
EPS = 1e-7      # perturbation size
THETA = 1e-4    # front-detection threshold on divergence


def divergence_field(lat, H0, T, perturb_site, eps=EPS):
    """Return D[t, i] = ||h_i^pert(t) - h_i^ref(t)|| for a point perturbation."""
    Hp = H0.copy()
    Hp[perturb_site] += eps * RNG.standard_normal(lat.m)
    ref = lat.run(H0, T)
    per = lat.run(Hp, T)
    D = np.linalg.norm(per - ref, axis=2)  # (T+1, n)
    return D


def front_positions(D, center, theta=THETA):
    """For each time, the max distance from center at which D>theta. Returns
    array of front radius vs time (NaN where no site exceeds threshold)."""
    T1, n = D.shape
    dist = np.abs(np.arange(n) - center)
    radius = np.full(T1, np.nan)
    for t in range(T1):
        active = dist[D[t] > theta]
        if active.size:
            radius[t] = active.max()
    return radius


def fit_velocity(radius, tmin=2, frac=0.6):
    """Fit slope (sites per step) of front radius vs time in the early-linear
    regime (before the front saturates near the boundary)."""
    valid = np.where(~np.isnan(radius))[0]
    valid = valid[valid >= tmin]
    if valid.size < 4:
        return np.nan
    rmax = np.nanmax(radius)
    use = valid[radius[valid] <= frac * rmax]
    if use.size < 4:
        use = valid[: max(4, int(frac * valid.size))]
    A = np.vstack([use, np.ones_like(use)]).T
    slope, _ = np.linalg.lstsq(A, radius[use], rcond=None)[0]
    return slope


# ----------------------------------------------------------------------
def E1_butterfly_vs_rho(rhos, n=241, m=24, T=120, coupling=0.18, seeds=6):
    """Homogeneous lattices: measure v_B at each rho."""
    center = n // 2
    vmean, vstd = [], []
    for rho in rhos:
        vs = []
        for s in range(seeds):
            lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=100 + s)
            H0 = 0.1 * RNG.standard_normal((n, m))
            D = divergence_field(lat, H0, T, center)
            r = front_positions(D, center)
            v = fit_velocity(r)
            if not np.isnan(v):
                vs.append(v)
        vmean.append(np.mean(vs) if vs else np.nan)
        vstd.append(np.std(vs) if vs else np.nan)
    return np.array(vmean), np.array(vstd)


def E2_trapping(n=241, m=24, T=160, coupling=0.18, rho_lo=0.6, rho_hi=2.2,
                band=40, leak_band=0.0, seed=3):
    """Central band of high rho (and optional saturation via leak). Inject at
    band centre; return divergence field and front radius vs time."""
    center = n // 2
    rho = np.full(n, rho_lo)
    rho[center - band: center + band + 1] = rho_hi
    lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, leak=0.0, seed=seed)
    if leak_band > 0:
        leakarr = np.zeros(n)
        leakarr[center - band: center + band + 1] = leak_band
        # rebuild with per-site leak: emulate by storing as attribute used in step
        lat.leak = leakarr  # step() supports array leak via broadcasting
    H0 = 0.1 * RNG.standard_normal((n, m))
    D = divergence_field(lat, H0, T, center)
    r = front_positions(D, center)
    return D, r, center, band


def E3_scrambling(sizes, m=24, coupling=0.18, rho=1.4, seeds=5):
    """Scrambling time: time for a central perturbation's influence to reach
    the lattice edge (front radius ~ N/2). t_* vs N; log N => fast scrambler."""
    tstar_mean, tstar_std = [], []
    for n in sizes:
        center = n // 2
        ts = []
        T = 6 * n  # allow enough time
        for s in range(seeds):
            lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=200 + s)
            H0 = 0.1 * RNG.standard_normal((n, m))
            D = divergence_field(lat, H0, T, center)
            r = front_positions(D, center)
            target = 0.9 * (n // 2)
            hit = np.where(r >= target)[0]
            if hit.size:
                ts.append(hit[0])
        tstar_mean.append(np.mean(ts) if ts else np.nan)
        tstar_std.append(np.std(ts) if ts else np.nan)
    return np.array(tstar_mean), np.array(tstar_std)


def E4_rule110_lightcone(n=401, T=200, seed=1):
    """Two random Rule-110 configs differing in one central cell; spread of
    the difference is the exact perturbation cone."""
    rng = np.random.default_rng(seed)
    row0 = rng.integers(0, 2, n).astype(np.uint8)
    row1 = row0.copy()
    row1[n // 2] ^= 1
    a = rule110_run(row0, T)
    b = rule110_run(row1, T)
    diff = (a ^ b).astype(np.uint8)
    center = n // 2
    dist = np.abs(np.arange(n) - center)
    radius = np.array([dist[diff[t] > 0].max() if diff[t].any() else 0
                       for t in range(T + 1)])
    # cone speed: slope of radius vs time
    A = np.vstack([np.arange(T + 1), np.ones(T + 1)]).T
    slope = np.linalg.lstsq(A, radius, rcond=None)[0][0]
    return diff, radius, slope, center


# ----------------------------------------------------------------------
def main():
    results = {}

    print("E1: butterfly velocity vs recurrence inertia rho ...")
    rhos = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0])
    v1, v1s = E1_butterfly_vs_rho(rhos)
    results["E1"] = {"rho": rhos.tolist(), "vB": v1.tolist(), "vB_std": v1s.tolist()}
    for r, v in zip(rhos, v1):
        print(f"    rho={r:4.1f}  v_B={v:.4f} sites/step")

    print("E2: trapping-surface test (high-rho band, and saturated band) ...")
    D_hi, r_hi, c2, band = E2_trapping(rho_hi=2.4, leak_band=0.0)
    D_sat, r_sat, _, _ = E2_trapping(rho_hi=1.2, leak_band=1.5)  # saturated band
    D_ctrl, r_ctrl, _, _ = E2_trapping(rho_hi=0.6, leak_band=0.0)  # uniform-ish control
    results["E2"] = {"band_halfwidth": band, "center": c2}

    print("E3: scrambling time vs system size ...")
    sizes = np.array([41, 61, 81, 121, 161, 201, 261])
    t3, t3s = E3_scrambling(sizes)
    results["E3"] = {"N": sizes.tolist(), "tstar": t3.tolist(), "tstar_std": t3s.tolist()}
    for nn, tt in zip(sizes, t3):
        print(f"    N={nn:4d}  t*={tt:.1f}")
    # fit linear (ballistic) vs log (fast scrambler)
    good = ~np.isnan(t3)
    lin = np.polyfit(sizes[good], t3[good], 1)
    logf = np.polyfit(np.log(sizes[good]), t3[good], 1)
    lin_res = np.sum((np.polyval(lin, sizes[good]) - t3[good]) ** 2)
    log_res = np.sum((np.polyval(logf, np.log(sizes[good])) - t3[good]) ** 2)
    results["E3"]["linear_fit_resid"] = float(lin_res)
    results["E3"]["log_fit_resid"] = float(log_res)
    results["E3"]["verdict"] = "ballistic" if lin_res < log_res else "log/fast-scrambler"
    print(f"    linear-fit resid={lin_res:.1f}  log-fit resid={log_res:.1f} -> {results['E3']['verdict']}")

    print("E4: Rule 110 exact light-cone ...")
    diff110, rad110, slope110, c4 = E4_rule110_lightcone()
    results["E4"] = {"cone_speed_sites_per_step": float(slope110)}
    print(f"    Rule 110 cone speed = {slope110:.3f} sites/step (LR velocity)")

    # ---------------- figures ----------------
    plt.figure(figsize=(7, 4.4))
    plt.errorbar(rhos, v1, yerr=v1s, marker="o", capsize=3)
    plt.axhline(0, color="k", lw=0.6)
    plt.xlabel(r"recurrence inertia  $\rho$ (local spectral radius)")
    plt.ylabel(r"butterfly velocity  $v_B$  (sites/step)")
    plt.title("E1: does recurrence inertia slow or speed information?")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_E1_butterfly_vs_rho.png", dpi=130)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for ax, D, title in [
        (axes[0], D_ctrl, r"control: ~uniform low $\rho$"),
        (axes[1], D_hi, r"high-$\rho$ band ($\rho=2.4$)"),
        (axes[2], D_sat, r"saturated band (leak$=1.5$)"),
    ]:
        Z = np.log10(D + 1e-16)
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="inferno",
                       vmin=-12, vmax=0)
        ax.axvline(c2 - band, color="cyan", lw=0.8, ls="--")
        ax.axvline(c2 + band, color="cyan", lw=0.8, ls="--")
        ax.set_xlabel("site")
        ax.set_ylabel("time step")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, label=r"$\log_{10}$ divergence", shrink=0.8)
    fig.suptitle("E2: trapping-surface test  (dashed = band edges; bright = perturbed)")
    plt.savefig("fig_E2_trapping.png", dpi=130)
    plt.close()

    plt.figure(figsize=(7, 4.4))
    plt.errorbar(sizes, t3, yerr=t3s, marker="s", capsize=3, label="measured $t_*$")
    xs = np.linspace(sizes.min(), sizes.max(), 100)
    plt.plot(xs, np.polyval(lin, xs), "r--", label="linear (ballistic) fit")
    plt.plot(xs, np.polyval(logf, np.log(xs)), "g--", label="log (fast-scrambler) fit")
    plt.xlabel("system size  N")
    plt.ylabel(r"scrambling time  $t_*$")
    plt.title("E3: scrambling-time scaling")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig_E3_scrambling.png", dpi=130)
    plt.close()

    plt.figure(figsize=(6.4, 5.2))
    plt.imshow(diff110, aspect="auto", origin="lower", cmap="binary",
               interpolation="nearest")
    plt.plot([c4, c4 + rad110[-1]], [0, len(rad110) - 1], "r-", lw=1,
             label=f"cone edge ~ {slope110:.2f} sites/step")
    plt.plot([c4, c4 - rad110[-1]], [0, len(rad110) - 1], "r-", lw=1)
    plt.xlabel("cell")
    plt.ylabel("time step")
    plt.title("E4: Rule 110 exact perturbation light-cone")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("fig_E4_rule110_cone.png", dpi=130)
    plt.close()

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWrote results.json and 4 figures.")
    return results


if __name__ == "__main__":
    main()
