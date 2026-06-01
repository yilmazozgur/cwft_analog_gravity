"""
The computational Clausius test -- TRUE 2D lattice, no radial reduction.

A genuine NxN reservoir lattice (full 2D nearest-neighbour coupling, no 1/r
approximation) with a radial damping well. The 2D geometry gives ~2*pi*r sites
per radius, so angular-averaging yields a clean radial gain profile g(r) -- the
cure for the kappa_c noise that limited the 1D reduction.

For each well we measure three INDEPENDENT functionals of the gain field:
  r_h  : innermost g(r)=0 crossing (core negative -> exterior positive)
  A    : horizon "area" = circumference = 2*pi*r_h        (true 2D: d=2)
  T_c  : kappa_c/(2pi),  kappa_c = (1/2)|dg/dr|_{r_h}      (hbar_c := 1)
  M_c  : computational mass = sum over interior grid cells of max(-g,0)
         (genuine 2D area integral; cell area = 1)

First law / Clausius: dM_c = T_c dS_c, S_c = eta_c A.
Test: eta_c = dM_c/(T_c dA) the SAME constant across a 2-parameter family
(varying well depth Lmax AND width w)?  Constant => Clausius holds =>
computational Einstein equation grounded, G_c = 1/(4 hbar_c eta_c).

Speed: local Jacobian spectral radii are computed BATCHED via np.linalg.eigvals
on a (N*N, m, m) array per sampled timestep.
"""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(0)
N, M, RHO, C = 61, 10, 2.6, 0.05      # grid, internal dim, recurrence, coupling
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

def gain_field(leak, T=240, burn=64, sample=8, seed=0):
    rng = np.random.default_rng(seed)
    H = 0.1 * rng.standard_normal((N, N, M))
    # M_base[i,j] = rho*W0 - leak[i,j] I   -> (N,N,m,m)
    Mbase = RHO * _W[None, None] - leak[..., None, None] * _EYE[None, None]
    acc = np.zeros((N, N)); cnt = 0
    for t in range(T):
        H, pre = step(H, leak)
        if t >= burn and (t - burn) % sample == 0:
            sech2 = 1.0 - np.tanh(pre) ** 2                # (N,N,m)
            B = sech2[..., :, None] * Mbase                # (N,N,m,m), row-scaled
            ev = np.linalg.eigvals(B.reshape(-1, M, M))    # batched
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

def measure(Lmax, w, seeds=2):
    leak = well(Lmax, w)
    gf = np.mean([gain_field(leak, seed=10 + s) for s in range(seeds)], axis=0)
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
    Mc = float(np.sum(np.clip(-gf, 0, None)[RAD < r_h]))   # genuine 2D interior integral
    return dict(Lmax=Lmax, w=w, r_h=float(r_h), A=float(A), kappa=float(kappa),
                Tc=float(Tc), Mc=Mc)

t0 = time.time()
print(f"Computational Clausius test -- TRUE 2D ({N}x{N}, m={M}):")
widths = [8.0, 11.0, 14.0, 17.0]
depths = [3.9, 4.5, 5.2, 5.9, 6.6, 7.4]
sweep = []
for w in widths:
    for Lmax in depths:
        m = measure(Lmax, w)
        if m:
            sweep.append(m)
            print(f"  w={w:4.1f} Lmax={Lmax:.1f}  r_h={m['r_h']:6.2f}  A={m['A']:7.1f}"
                  f"  T_c={m['Tc']:.4f}  M_c={m['Mc']:8.1f}")
print(f"  ({len(sweep)} black holes; {time.time()-t0:.0f}s)")

# mass-area scaling
allA = np.array([m["A"] for m in sweep]); allM = np.array([m["Mc"] for m in sweep])
p_fit = np.polyfit(np.log(allA), np.log(allM), 1)[0]
print(f"\n  mass-area scaling:  M_c ~ A^{p_fit:.2f}")

# first-law test, well-formed horizons
def collect(rh_min):
    etas, pts = [], []
    for w in widths:
        sl = sorted([m for m in sweep if m["w"] == w and m["r_h"] >= rh_min],
                    key=lambda d: d["A"])
        for a, b in zip(sl[:-1], sl[1:]):
            dA = b["A"] - a["A"]; dM = b["Mc"] - a["Mc"]; Tc = 0.5 * (a["Tc"] + b["Tc"])
            if abs(dA) > 1e-3:
                etas.append(dM / (Tc * dA)); pts.append((Tc * dA, dM))
    return np.array(etas), np.array(pts)

for tag, rhmin in [("ALL", 0.0), ("well-formed r_h>=6", 6.0)]:
    e, p = collect(rhmin)
    if e.size >= 2:
        cv = np.std(e) / (abs(np.mean(e)) + 1e-12)
        x, y = p[:, 0], p[:, 1]
        s = np.sum(x * y) / np.sum(x * x)
        r2 = 1 - np.sum((y - s * x) ** 2) / np.sum((y - y.mean()) ** 2)
        print(f"  [{tag}] eta_c median={np.median(e):.1f} CV={cv:.2f} "
              f"fit slope={s:.1f} R^2={r2:.2f} (n={e.size})")

print("  per-slice eta_c (well-formed):")
for w in widths:
    sl = sorted([m for m in sweep if m["w"] == w and m["r_h"] >= 5.0], key=lambda d: d["A"])
    es = [(b["Mc"] - a["Mc"]) / (0.5 * (a["Tc"] + b["Tc"]) * (b["A"] - a["A"]))
          for a, b in zip(sl[:-1], sl[1:]) if abs(b["A"] - a["A"]) > 1e-3]
    if es:
        print(f"    w={w:4.1f}: eta_c={np.median(es):.1f} (n={len(es)})")

e, p = collect(5.0)
eta_med = float(np.median(e)); eta_cv = float(np.std(e) / (abs(np.mean(e)) + 1e-12))
x, y = p[:, 0], p[:, 1]
slope_fit = float(np.sum(x * y) / np.sum(x * x))
r2 = float(1 - np.sum((y - slope_fit * x) ** 2) / np.sum((y - y.mean()) ** 2))
G_c = 1.0 / (4 * eta_med) if eta_med else None
if eta_cv < 0.25 and r2 > 0.85:
    verdict = "CLAUSIUS HOLDS (near-universal eta_c) -> computational Einstein eq. grounded"
elif eta_cv < 0.5:
    verdict = "PARTIAL: approximate first law; not clean enough to ground Einstein dynamics"
else:
    verdict = "CLAUSIUS FAILS: analog kinematics without Einstein dynamics"
print(f"\n  G_c = 1/(4 hbar_c eta_c) = {G_c:.4g}")
print(f"  mass-area exponent p = {p_fit:.2f}")
print(f"  VERDICT: {verdict}")

res = json.load(open("results.json"))
res["ClausiusTest2D"] = {"grid": N, "m": M, "rho": RHO, "coupling": C,
                         "n_blackholes": len(sweep), "hbar_c": 1.0,
                         "mass_area_exponent_p": float(p_fit),
                         "eta_median": eta_med, "eta_cv": eta_cv,
                         "clausius_fit_slope": slope_fit, "clausius_fit_R2": r2,
                         "G_c": G_c, "verdict": verdict}
json.dump(res, open("results.json", "w"), indent=2)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.7))
for w, col in zip(widths, ["C0", "C1", "C2", "C3"]):
    sl = sorted([m for m in sweep if m["w"] == w], key=lambda d: d["A"])
    a1.plot([m["A"] for m in sl], [m["Mc"] for m in sl], "o-", color=col, label=f"w={w:.0f}")
a1.set_xlabel("horizon area $A=2\\pi r_h$"); a1.set_ylabel("computational mass $M_c$")
a1.set_xscale("log"); a1.set_yscale("log")
a1.set_title(f"TRUE 2D: $M_c\\sim A^{{{p_fit:.2f}}}$"); a1.legend(); a1.grid(alpha=0.3, which="both")
a2.plot(x, y, "o", ms=6, alpha=0.8)
xl = np.linspace(0, x.max() * 1.05, 50)
a2.plot(xl, slope_fit * xl, "r-", label=f"$dM_c=\\eta_c T_c dA$\n$\\eta_c$={slope_fit:.0f} $R^2$={r2:.2f} CV={eta_cv:.2f}")
a2.set_xlabel("$T_c\\,dA$"); a2.set_ylabel("$dM_c$"); a2.grid(alpha=0.3)
a2.set_title("Clausius first-law test (true 2D)"); a2.legend()
plt.tight_layout(); plt.savefig("fig_Clausius_2D.png", dpi=300); plt.close()
print("Wrote fig_Clausius_2D.png")
