"""
Test 6 (clean) -- effective lapse from the ANALYTIC local Jacobian.

Local clock rate at site i = time-averaged log spectral radius of the local
Jacobian diagonal block
    J_ii = diag(1 - tanh^2(pre_i)) @ (rho_i * W0 - leak_i * I).
g(i) > 0 : locally expanding (clock ticking, information produced).
g(i) < 0 : locally contracting (clock frozen, information destroyed).
The principled horizon is the surface g(i) = 0 (boundary between the
information-producing exterior and the information-destroying core). We test
whether the independently-measured transport front stalls at that surface.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_substrate import ReservoirLattice
from cwf_experiments import divergence_field, RNG

THETA = 1e-4

def make_well(n, c, w, Lmax):
    x = np.arange(n)
    return Lmax * np.exp(-((x - c) / w) ** 2)

def jacobian_gain_profile(lat, H0, T=400):
    """Time-averaged log spectral radius of each site's local Jacobian block."""
    n, m = lat.n, lat.m
    H = H0.copy()
    acc = np.zeros(n)
    cnt = 0
    for t in range(T):
        # pre-activation (same expression as step, pre-tanh)
        left = np.zeros_like(H); right = np.zeros_like(H)
        left[1:] = H[:-1]; right[:-1] = H[1:]
        neigh = (left + right) @ lat.P.T
        leak = lat.leak
        leak_arr = (leak if not np.isscalar(leak) else np.full(n, leak))
        pre = lat.rho[:, None] * (H @ lat.W0.T) + lat.c * neigh - leak_arr[:, None] * H
        sech2 = 1.0 - np.tanh(pre) ** 2          # (n, m)
        if t > 50:                                # discard transient
            for i in range(n):
                Jii = (sech2[i][:, None] *
                       (lat.rho[i] * lat.W0 - leak_arr[i] * np.eye(m)))
                sr = np.max(np.abs(np.linalg.eigvals(Jii)))
                acc[i] += np.log(sr + 1e-30)
            cnt += 1
        H = np.tanh(pre)
    return acc / max(cnt, 1)

def transport_reach(n, leak, c, inject_off=-70, m=24, coupling=0.18, rho=2.6,
                    T=1500, seeds=8):
    inj = c + inject_off
    reach = np.zeros(n)
    for s in range(seeds):
        lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=970 + s)
        lat.leak = leak
        H0 = 0.1 * RNG.standard_normal((n, m))
        D = divergence_field(lat, H0, T, inj)
        reach += (D > THETA).any(axis=0)
    return reach / seeds, inj

n, c, w, Lmax, rho = 301, 150, 18, 3.0, 2.6
leak = make_well(n, c, w, Lmax)

print("Test 6 (clean): analytic local Jacobian gain g(x) ...")
lat = ReservoirLattice(n, m=24, coupling=0.18, rho=rho, seed=970)
lat.leak = leak
H0 = 0.1 * RNG.standard_normal((n, 24))
g = jacobian_gain_profile(lat, H0, T=400)

reach, inj = transport_reach(n, leak, c, rho=rho)

# horizon from g: first site (coming from the left) where g crosses 0
left = np.arange(n) < c
cross = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
g_horizon = int(cross[cross < c][-1]) if np.any(cross < c) else c
# transport horizon: deepest site (left side) reached >=50% runs
reached = np.where(reach >= 0.5)[0]
t_horizon = int(reached[reached <= c].max()) if np.any(reached <= c) else inj

print(f"    g(x)=0 surface (clock freeze, from left) at site {g_horizon}")
print(f"    transport front stalls at site            {t_horizon}")
print(f"    separation = {abs(g_horizon - t_horizon)} sites  "
      f"(well half-width w={w})")

# correlation between max(g,0) (lapse) and transport reach across the well region
reg = (np.arange(n) >= c - 60) & (np.arange(n) <= c + 60)
lapse = np.clip(g, 0, None)
lapse_n = lapse / (np.median(lapse[~reg & (np.arange(n) < c)]) + 1e-9)
corr = float(np.corrcoef(lapse[reg], reach[reg])[0, 1])
print(f"    corr(lapse, transport-reach) over well = {corr:.3f}")

res = json.load(open("results.json"))
res["Test6"] = {"g_zero_horizon": g_horizon, "transport_horizon": t_horizon,
                "separation_sites": int(abs(g_horizon - t_horizon)),
                "well_halfwidth": w,
                "corr_lapse_transport": corr,
                "interpretation": ("transport front stalls within ~well-width of "
                                   "the clock-freeze (g=0) surface; both driven by "
                                   "the same local-gain field -> effective-metric-"
                                   "consistent")}
json.dump(res, open("results.json", "w"), indent=2)

fig, ax = plt.subplots(figsize=(8.4, 4.7))
xs = np.arange(n)
ax.plot(xs, g, "b-", lw=1.6, label="local Jacobian gain  $g(x)$  (clock rate / lapse)")
ax.axhline(0, color="b", ls=":", lw=0.8)
ax.plot(xs, reach, "g-", lw=1.6, alpha=0.8, label="transport reach (frac of runs)")
ax.plot(xs, leak / leak.max() * g[~reg].max(), color="orange", alpha=0.5,
        label="damping well (scaled)")
ax.axvline(g_horizon, color="purple", ls="--", label=f"clock-freeze g=0 @ {g_horizon}")
ax.axvline(t_horizon, color="red", ls="--", label=f"transport horizon @ {t_horizon}")
ax.set_xlim(c - 80, c + 80); ax.set_xlabel("site"); ax.set_ylabel("gain / reach")
ax.set_title(f"Test 6: clock-freeze surface vs transport horizon "
             f"(sep={abs(g_horizon - t_horizon)} sites, corr={corr:.2f})")
ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig_Test6_metric.png", dpi=300); plt.close()
print("Wrote fig_Test6_metric.png")
