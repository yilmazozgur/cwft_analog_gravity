"""
A2 phase 1 -- power-law non-locality and the onset of fast scrambling.

A 2D reservoir lattice with translation-invariant power-law spatial coupling
    K(r) propto |r|^{-alpha}   (toroidal distance; K(0)=0)
applied via FFT-based convolution. Substrate parameters (rho=2.6, m=10,
coupling=0.05) match cwf_clausius2d.py so that phase 2 (Clausius rerun
inside a fast-scrambling regime) plugs in without rebuilding.

Normalization: K is rescaled so that sum(K)=4. That matches NN's 4-neighbour
total in the alpha->infinity limit -- so the alpha sweep is a pure structural
change (which sites couple to which), not a change in overall coupling
strength. We verify the alpha->inf limit recovers exact NN below.

t* protocol (full-saturation, fast-scrambler-canonical):
    perturb one channel at the center site by eps=1e-7;
    run ref + perturbed trajectories;
    t* = first step at which |H_per - H_ref|_2 > theta=1e-4 at EVERY site.

Expected scaling:
    alpha large (>> d=2):  K concentrates on NN; t* ~ N           (ballistic)
    alpha small (<~ d=2):  K's tail dominates;  t* ~ log N         (fast scrambler)

Phase 1 only validates the substrate. Phase 2 will rerun the Clausius/
Einstein test inside a regime where this experiment confirms fast scrambling.
"""

import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Substrate parameters chosen so the NN limit is robustly CHAOTIC.
# Note: this DIFFERS from cwf_clausius2d.py's c=0.05, which we discovered is
# globally contractive (lambda ~ -0.2) despite having positive local gain
# g(x) > 0 -- the local-Jacobian spectral radius (what gain_field measures)
# is NOT the full-dynamics Lyapunov exponent. Phase 2 will rebaseline the
# Clausius test at c=0.50 before adding non-locality.
M, RHO, C = 10, 2.6, 0.50  # NN limit: lambda ~ +0.18, growth time ~38 steps
_RNG_INIT = np.random.default_rng(0)
_W = _RNG_INIT.standard_normal((M, M)); _W /= max(abs(np.linalg.eigvals(_W)))
_P = _RNG_INIT.standard_normal((M, M)); _P /= max(abs(np.linalg.eigvals(_P)))


# -------- power-law kernel --------------------------------------------------
def make_kernel(N, alpha):
    """Translation-invariant power-law kernel on a NxN torus.

    Returns K of shape (N, N) in FFT-natural ordering (zero offset at index 0,
    +k at index k for k <= N//2, -k at index N-k). K(0,0)=0 (self-exclusion);
    K(i,j) = 1/r^alpha for r = sqrt(dx^2 + dy^2) > 0 with toroidal distances
    dx, dy. Sum-normalized so sum(K)=4 (matching the NN 4-neighbour total in
    the alpha->infinity limit)."""
    idx = np.arange(N)
    idx = np.where(idx <= N // 2, idx, idx - N)
    I, J = np.meshgrid(idx, idx, indexing="ij")
    r = np.sqrt(I.astype(float) ** 2 + J.astype(float) ** 2)
    K = np.zeros_like(r)
    nz = r > 0
    K[nz] = 1.0 / (r[nz] ** alpha)
    K *= 4.0 / K.sum()
    return K


# -------- FFT-based step ----------------------------------------------------
def step_pl(H, leak, K_fft, out_shape):
    """One synchronous update of the 2D power-law reservoir.
       H: (N, N, m).   leak: (N, N) array of per-site damping.
       K_fft: precomputed np.fft.rfft2(K) of shape (N, N//2+1)."""
    # spatial convolution: rfft over axes (0, 1)
    Hf = np.fft.rfft2(H, axes=(0, 1))
    nb = np.fft.irfft2(Hf * K_fft[..., None], s=out_shape, axes=(0, 1))
    pre = RHO * (H @ _W.T) + C * (nb @ _P.T) - leak[..., None] * H
    return np.tanh(pre)


# -------- scrambling-time measurement --------------------------------------
def t_scramble(N, alpha, eps=1e-7, theta=1e-4, warmup=200, T_max=None,
               seeds=6, rng_base=42, verbose=False):
    """Median t* across seeds; full-saturation criterion.
    Perturbation is a random m-vector (not single channel) -- matches the
    cwf_experiments protocol and excites all internal modes."""
    if T_max is None:
        T_max = max(12 * N, 400)
    K = make_kernel(N, alpha)
    K_fft = np.fft.rfft2(K)
    no_leak = np.zeros((N, N))
    out_shape = (N, N)
    center = N // 2
    vals = []
    for s in range(seeds):
        rng = np.random.default_rng(rng_base + s)
        H = 0.1 * rng.standard_normal((N, N, M))
        # warmup
        for _ in range(warmup):
            H = step_pl(H, no_leak, K_fft, out_shape)
        # perturb with random m-vector at center
        Hp = H.copy()
        Hp[center, center] += eps * rng.standard_normal(M)
        ref = H; per = Hp
        tstar = None
        for t in range(1, T_max + 1):
            ref = step_pl(ref, no_leak, K_fft, out_shape)
            per = step_pl(per, no_leak, K_fft, out_shape)
            D = np.linalg.norm(per - ref, axis=2)   # (N, N)
            if D.min() > theta:
                tstar = t
                break
        if tstar is not None:
            vals.append(tstar)
        if verbose:
            print(f"      seed {s}: t*={tstar}")
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.median(vals)), float(np.std(vals)), len(vals)


# -------- sanity: verify alpha->inf recovers NN ---------------------------
def sanity_check_NN_limit():
    """For alpha=8, the kernel should match NN to within ~1%. Verify step
    output matches np.roll-based step on a random H."""
    N = 21
    alpha = 8.0
    K = make_kernel(N, alpha)
    K_fft = np.fft.rfft2(K)
    rng = np.random.default_rng(7)
    H = 0.1 * rng.standard_normal((N, N, M))
    leak = np.zeros((N, N))
    # FFT step
    Hf = np.fft.rfft2(H, axes=(0, 1))
    nb_fft = np.fft.irfft2(Hf * K_fft[..., None], s=(N, N), axes=(0, 1))
    # NN step
    nb_nn = (np.roll(H, 1, 0) + np.roll(H, -1, 0)
             + np.roll(H, 1, 1) + np.roll(H, -1, 1))
    err = np.max(np.abs(nb_fft - nb_nn))
    rel = err / (np.max(np.abs(nb_nn)) + 1e-12)
    print(f"  Sanity: alpha=8 vs NN  max abs err = {err:.3e}  rel = {rel:.3e}")
    # also alpha=12 -> should be tighter
    K2 = make_kernel(N, 12.0)
    K2_fft = np.fft.rfft2(K2)
    Hf2 = np.fft.rfft2(H, axes=(0, 1))
    nb2 = np.fft.irfft2(Hf2 * K2_fft[..., None], s=(N, N), axes=(0, 1))
    err2 = np.max(np.abs(nb2 - nb_nn))
    rel2 = err2 / (np.max(np.abs(nb_nn)) + 1e-12)
    print(f"  Sanity: alpha=12 vs NN max abs err = {err2:.3e} rel = {rel2:.3e}")


# -------- classification ----------------------------------------------------
def classify(N_arr, t_arr):
    """Classify t*(N) as constant / log / ballistic / intermediate.

    Constant ('fast scrambler'): t* essentially flat with N (slope -> 0).
    Log: t* ~ a + b log N with b > 0.
    Ballistic: t* ~ a + b N with b > 0.
    Compares residuals against a constant-mean baseline."""
    good = ~np.isnan(t_arr)
    if good.sum() < 3:
        return "insufficient", None, None, None
    x = N_arr[good].astype(float); y = t_arr[good].astype(float)
    # baseline: constant fit
    y_mean = y.mean()
    r_const = float(np.sum((y - y_mean) ** 2))
    # linear fit
    lin = np.polyfit(x, y, 1)
    r_lin = float(np.sum((y - np.polyval(lin, x)) ** 2))
    # log fit
    log = np.polyfit(np.log(x), y, 1)
    r_log = float(np.sum((y - np.polyval(log, np.log(x))) ** 2))
    # slope ranges of x in actual units
    Nspan = x.max() - x.min()
    lin_growth = lin[0] * Nspan          # expected t* growth across N range
    log_growth = log[0] * np.log(x.max() / x.min())
    yspan = max(y.max() - y.min(), 1.0)
    # if neither linear nor log explains the spread, call it constant
    # (fast-scrambler regime: t* essentially N-independent)
    rel_lin = (r_const - r_lin) / r_const
    rel_log = (r_const - r_log) / r_const
    explained = max(rel_lin, rel_log)
    if explained < 0.3:
        v = "constant (fast scrambler: t* ~ N-independent)"
    elif r_log < 0.5 * r_lin and log_growth > 0:
        v = "log (~log N)"
    elif r_lin < 0.5 * r_log and lin_growth > 0:
        v = "ballistic (~N)"
    elif r_log < r_lin:
        v = "log-leaning intermediate"
    else:
        v = "ballistic-leaning intermediate"
    return v, r_lin, r_log, r_const


# -------- per-alpha global Lyapunov diagnostic ------------------------------
def lyapunov_alpha(N, alpha, warmup=200, T=80, eps=1e-7, seeds=4):
    """Global-norm Lyapunov from a single-site perturbation. Tells us whether
    the small-alpha 'no saturation' regime is contractive (lambda < 0) or
    just synchronization (lambda > 0 longitudinally but transverse damps)."""
    K = make_kernel(N, alpha)
    K_fft = np.fft.rfft2(K)
    no_leak = np.zeros((N, N))
    out_shape = (N, N)
    c = N // 2
    slopes = []
    for s in range(seeds):
        rng = np.random.default_rng(2000 + s)
        H = 0.1 * rng.standard_normal((N, N, M))
        for _ in range(warmup):
            H = step_pl(H, no_leak, K_fft, out_shape)
        Hp = H.copy(); Hp[c, c] += eps * rng.standard_normal(M)
        ref, per = H, Hp
        D_traj = []
        for t in range(T):
            ref = step_pl(ref, no_leak, K_fft, out_shape)
            per = step_pl(per, no_leak, K_fft, out_shape)
            D_traj.append(float(np.linalg.norm(per - ref)))
        D_traj = np.array(D_traj)
        t_fit = np.arange(5, min(T, 35))
        slope = float(np.polyfit(t_fit, np.log(D_traj[t_fit] + 1e-30), 1)[0])
        slopes.append(slope)
    return float(np.mean(slopes)), float(np.std(slopes))


# -------- main driver -------------------------------------------------------
def main():
    sanity_check_NN_limit()
    print()

    Ns = np.array([21, 31, 41, 61, 81, 121, 161])
    alphas = [6.0, 5.0, 4.0, 3.5, 3.0, 2.75, 2.5, 2.25, 2.0, 1.5, 1.0, 0.5]
    seeds = 6
    t0 = time.time()
    print(f"A2 phase 1: power-law scrambling sweep")
    print(f"  alphas={alphas}")
    print(f"  Ns={Ns.tolist()}, seeds={seeds}\n")

    by_alpha = {}
    for alpha in alphas:
        ts, sds, ns_ok = [], [], []
        ta = time.time()
        for N in Ns:
            tm, ts_, n_ok = t_scramble(int(N), alpha, seeds=seeds)
            ts.append(tm); sds.append(ts_); ns_ok.append(n_ok)
            print(f"  alpha={alpha:4.2f}  N={int(N):3d}  "
                  f"t*={tm:7.1f} +/- {ts_:6.1f}  (n_ok={n_ok}/{seeds})")
        verdict, r_lin, r_log, r_const = classify(Ns, np.array(ts))
        # per-alpha global Lyapunov diagnostic (at N=41, cheap)
        lam, lam_std = lyapunov_alpha(41, alpha)
        dt = time.time() - ta
        by_alpha[f"alpha_{alpha}"] = dict(
            alpha=alpha, N=Ns.tolist(),
            t_star_median=ts, t_star_std=sds,
            n_ok=ns_ok, verdict=verdict,
            resid_linear=r_lin, resid_log=r_log, resid_const=r_const,
            lyapunov_global=lam, lyapunov_std=lam_std,
            elapsed_s=dt,
        )
        print(f"  -> {verdict}  |  lambda(global) = {lam:+.3f}+-{lam_std:.3f}"
              f"  ({dt:.0f}s)\n")
    print(f"  total runtime {time.time() - t0:.0f}s")

    # save
    out = os.path.join(os.path.dirname(__file__) or ".", "results.json")
    r_all = json.load(open(out)) if os.path.exists(out) else {}
    r_all["A2_phase1_powerlaw_scrambling"] = dict(
        substrate="2D power-law reservoir (FFT-convolution coupling)",
        m=M, rho=RHO, coupling=C, eps=1e-7, theta=1e-4, warmup=200,
        kernel_normalization="sum(K)=4 (matches NN at alpha->inf)",
        N_grid=Ns.tolist(), alphas=alphas, seeds=seeds,
        by_alpha=by_alpha,
    )
    json.dump(r_all, open(out, "w"), indent=2)

    # plots --------------------------------------------------------------
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)
    ax_lin = fig.add_subplot(gs[0, 0])
    ax_lN  = fig.add_subplot(gs[0, 1])
    ax_ll  = fig.add_subplot(gs[0, 2])
    ax_lam = fig.add_subplot(gs[1, 0])
    ax_t41 = fig.add_subplot(gs[1, 1])
    ax_nok = fig.add_subplot(gs[1, 2])

    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(alphas)))
    for alpha, col in zip(alphas, colors):
        r = by_alpha[f"alpha_{alpha}"]
        ts = np.array(r["t_star_median"], float)
        sds = np.array(r["t_star_std"], float)
        good = ~np.isnan(ts)
        if good.sum() == 0:
            continue
        ax_lin.errorbar(Ns[good], ts[good], yerr=sds[good], fmt="o-",
                        color=col, capsize=3, lw=1.4, ms=4,
                        label=f"alpha={alpha}")
        ax_lN.errorbar(Ns[good], ts[good], yerr=sds[good], fmt="o-",
                       color=col, capsize=3, lw=1.4, ms=4)
        ax_ll.errorbar(Ns[good], ts[good], yerr=sds[good], fmt="o-",
                       color=col, capsize=3, lw=1.4, ms=4)
    ax_lin.set_xlabel("N"); ax_lin.set_ylabel(r"$t_*$")
    ax_lin.set_title("(a) Linear")
    ax_lin.grid(alpha=0.3); ax_lin.legend(fontsize=7, ncol=2)
    ax_lN.set_xscale("log")
    ax_lN.set_xlabel("N (log)"); ax_lN.set_ylabel(r"$t_*$")
    ax_lN.set_title(r"(b) $t_*$ vs $\log N$")
    ax_lN.grid(alpha=0.3, which="both")
    ax_ll.set_xscale("log"); ax_ll.set_yscale("log")
    ax_ll.set_xlabel("N (log)"); ax_ll.set_ylabel(r"$t_*$ (log)")
    ax_ll.set_title(r"(c) log-log")
    ax_ll.grid(alpha=0.3, which="both")

    # (d) lambda(alpha) -- the key diagnostic
    alpha_arr = np.array([by_alpha[f"alpha_{a}"]["alpha"] for a in alphas])
    lam_arr   = np.array([by_alpha[f"alpha_{a}"]["lyapunov_global"] for a in alphas])
    lam_std   = np.array([by_alpha[f"alpha_{a}"]["lyapunov_std"] for a in alphas])
    ax_lam.errorbar(alpha_arr, lam_arr, yerr=lam_std, fmt="o-", color="C3",
                    capsize=3, lw=1.5)
    ax_lam.axhline(0, color="k", lw=0.5, ls=":")
    ax_lam.set_xlabel(r"$\alpha$"); ax_lam.set_ylabel(r"global Lyapunov $\lambda$")
    ax_lam.set_title("(d) Transverse Lyapunov vs alpha (N=41)")
    ax_lam.grid(alpha=0.3)
    ax_lam.invert_xaxis()  # more non-local -> right

    # (e) t* at fixed N=41 vs alpha -- visualizes the optimal alpha
    t41 = []
    for a in alphas:
        r = by_alpha[f"alpha_{a}"]
        try:
            idx = list(Ns).index(41)
            t41.append(r["t_star_median"][idx])
        except (ValueError, IndexError):
            t41.append(float("nan"))
    t41 = np.array(t41)
    ax_t41.plot(alpha_arr, t41, "s-", color="C2", lw=1.5, ms=6)
    ax_t41.set_xlabel(r"$\alpha$"); ax_t41.set_ylabel(r"$t_*$ at N=41")
    ax_t41.set_title("(e) Scrambling time vs alpha (N=41)")
    ax_t41.grid(alpha=0.3); ax_t41.invert_xaxis()

    # (f) success fraction vs alpha (largest N)
    nok_lastN = np.array([by_alpha[f"alpha_{a}"]["n_ok"][-1] for a in alphas]) / seeds
    nok_firstN = np.array([by_alpha[f"alpha_{a}"]["n_ok"][0] for a in alphas]) / seeds
    ax_nok.plot(alpha_arr, nok_firstN, "o-", color="C0", lw=1.5,
                label=f"N={int(Ns[0])}")
    ax_nok.plot(alpha_arr, nok_lastN, "s-", color="C1", lw=1.5,
                label=f"N={int(Ns[-1])}")
    ax_nok.set_xlabel(r"$\alpha$"); ax_nok.set_ylabel("scrambling success rate")
    ax_nok.set_title("(f) Fraction of seeds that saturate")
    ax_nok.grid(alpha=0.3); ax_nok.invert_xaxis(); ax_nok.legend(fontsize=8)

    plt.suptitle("A2 phase 1: power-law non-locality and the scrambling regime"
                 "  (m=10, c=0.50, rho=2.6)",
                 fontsize=11.5)
    fig_path = os.path.join(os.path.dirname(__file__) or ".",
                            "fig_A2_phase1_scrambling.png")
    plt.savefig(fig_path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"\nWrote {fig_path}")


if __name__ == "__main__":
    main()
