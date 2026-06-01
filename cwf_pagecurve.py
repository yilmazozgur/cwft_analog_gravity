"""
Test 3 -- the Page curve via Clifford (stabilizer) scrambling dynamics.

A random Clifford circuit scrambles N qubits into a near-maximally-entangled
stabilizer state. Entanglement entropy of a contiguous region A of size L
follows the Page curve: S(L) rises ~linearly, peaks at N/2, falls symmetrically.
Read as black-hole evaporation: revealing qubits one at a time into 'radiation',
S(radiation) rises then falls -- information returns after the Page time.

Entanglement entropy of a stabilizer state (in bits):
    S_A = rank_GF2(G|_B) - |B|,
where G is the stabilizer-generator matrix in symplectic [X|Z] form and G|_B is
its restriction to the qubits of B = complement of A.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import stim


def gf2_rank(M):
    M = M.copy().astype(np.uint8) & 1
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i; break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
        if r == rows:
            break
    return r


def stabilizer_matrix(sim, N):
    """N x 2N binary symplectic matrix [X | Z] of the stabilizer generators."""
    stabs = sim.canonical_stabilizers()
    X = np.zeros((N, N), np.uint8); Z = np.zeros((N, N), np.uint8)
    for i, ps in enumerate(stabs):
        xs, zs = ps.to_numpy()
        X[i] = xs.astype(np.uint8); Z[i] = zs.astype(np.uint8)
    return X, Z


def entropy_region(X, Z, A_qubits, N):
    """Entanglement entropy (bits) of region A for a stabilizer state."""
    B = np.array([q for q in range(N) if q not in set(A_qubits)])
    if B.size == 0:
        return 0.0
    GB = np.hstack([X[:, B], Z[:, B]])     # generators restricted to B
    return gf2_rank(GB) - B.size


def random_clifford_state(N, depth, seed):
    rng = np.random.default_rng(seed)
    sim = stim.TableauSimulator(seed=int(seed))
    for d in range(depth):
        offset = d % 2
        pairs = [(i, i + 1) for i in range(offset, N - 1, 2)]
        for (a, b) in pairs:
            t = stim.Tableau.random(2)
            sim.do_tableau(t, [a, b])
    return sim


def page_curve(N=40, depth=60, seeds=8):
    Ls = np.arange(0, N + 1)
    acc = np.zeros(len(Ls))
    for s in range(seeds):
        sim = random_clifford_state(N, depth, seed=10 + s)
        X, Z = stabilizer_matrix(sim, N)
        for k, L in enumerate(Ls):
            acc[k] += entropy_region(X, Z, list(range(L)), N)
    return Ls, acc / seeds


print("Test 3: Clifford Page curve ...")
Ls, S = page_curve(N=40, depth=60, seeds=8)
peak_L = int(Ls[np.argmax(S)])
print(f"    N=40: S peaks at L={peak_L} (Page prediction N/2=20), S_max={S.max():.2f} bits")
# Page-value comparison: ideal random-state S(L) ~ min(L, N-L) - small correction
ideal = np.minimum(Ls, 40 - Ls)
print(f"    S(N/2)={S[20]:.2f} bits  (max possible = N/2 = 20)")

res = json.load(open("results.json"))
res["Test3_PageCurve"] = {"N": 40, "depth": 60, "L": Ls.tolist(),
                          "entropy_bits": S.tolist(), "peak_L": peak_L,
                          "S_at_half": float(S[20]),
                          "interpretation": ("scrambling produces the Page curve: "
                                             "S(L) rises, peaks at N/2, falls -- "
                                             "near-maximal entanglement, the "
                                             "black-hole evaporation signature")}
json.dump(res, open("results.json", "w"), indent=2)

plt.figure(figsize=(7.2, 4.6))
plt.plot(Ls, S, "o-", ms=4, label="Clifford scrambled state")
plt.plot(Ls, ideal, "k--", alpha=0.5, label="Page ideal  min(L, N$-$L)")
plt.axvline(20, color="r", ls=":", alpha=0.6, label="Page time (N/2)")
plt.xlabel("subsystem size  L  (qubits revealed as 'radiation')")
plt.ylabel("entanglement entropy  S  (bits)")
plt.title("Test 3: Page curve from Clifford scrambling (N=40)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("fig_Test3_pagecurve.png", dpi=130); plt.close()
print("Wrote fig_Test3_pagecurve.png")
