"""
Test 5 (redone) -- the unconstructability ladder as a transmission coefficient.

Fast-propagating background (rho=2.6), inject left of a thin central band,
detect right. Sweep band damping continuously; transmission coefficient =
fraction of runs whose front crosses to the detector. This is the localized
observability O: it falls from 1 (constructable) to 0 (unconstructable) as
transport is suppressed -- the unconstructability ladder.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cwf_substrate import ReservoirLattice
from cwf_experiments import divergence_field, RNG

THETA = 1e-4

def transmission(leak_band, n=261, m=24, coupling=0.18, rho=2.6, band=10,
                 inject_off=-45, detect_off=45, T=1400, seeds=12):
    center = n // 2
    inj, det = center + inject_off, center + detect_off
    leak = np.zeros(n)
    leak[center - band: center + band + 1] = leak_band
    crossed = 0; cross_times = []
    for s in range(seeds):
        lat = ReservoirLattice(n, m=m, coupling=coupling, rho=rho, seed=1500 + s)
        lat.leak = leak
        H0 = 0.1 * RNG.standard_normal((n, m))
        D = divergence_field(lat, H0, T, inj)
        hit = np.where(D[:, det] > THETA)[0]
        if hit.size:
            crossed += 1; cross_times.append(int(hit[0]))
    return crossed / seeds, (np.median(cross_times) if cross_times else None)

print("Test 5 (ladder): transmission coefficient vs band damping:")
leaks = [0.0, 0.5, 1.0, 1.4, 1.8, 2.2, 2.6, 3.2, 4.0]
trans, ctimes = [], []
for L in leaks:
    tr, ct = transmission(L)
    trans.append(tr); ctimes.append(ct)
    print(f"    leak={L:4.1f}  transmission O={tr:.2f}  cross_time={ct}")
trans = np.array(trans)

res = json.load(open("results.json"))
res["Test5"] = {"leak": leaks, "transmission_O": trans.tolist(), "cross_time": ctimes,
                "interpretation": ("transmission (localized observability O) falls "
                                   "from 1 (constructable) to 0 (unconstructable) as "
                                   "transport suppression rises -- the horizon closes "
                                   "at a critical damping")}
json.dump(res, open("results.json", "w"), indent=2)

plt.figure(figsize=(7.2, 4.4))
plt.plot(leaks, trans, "o-", ms=6)
plt.axhline(0.0, color="r", ls="--", alpha=0.6, label="unconstructable (horizon closed)")
plt.axhline(1.0, color="g", ls="--", alpha=0.6, label="constructable (transparent)")
plt.xlabel("band damping  (transport suppression)")
plt.ylabel("transmission coefficient  $O$  (frac. crossing)")
plt.title("Test 5: unconstructability ladder")
plt.ylim(-0.05, 1.05); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("fig_Test5_reconstruction.png", dpi=130); plt.close()
print("Wrote fig_Test5_reconstruction.png")
