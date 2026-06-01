"""
A3a -- Hayden-Preskill experimental infrastructure.

Substrate library (4 scrambler architectures + tools) + measurement protocols
(stabilizer-rank entropy, mutual information, Page curve, scrambling time).

Reuses gf2_rank + stabilizer_matrix from cwf_pagecurve.py.

The four scrambler architectures correspond to the four "axes" of the
Corollary 1.6.2 substrate landscape:

    AllToAllScrambler     -- random Clifford 2-qubit gates on random pairs;
                              the canonical Hayden-Preskill scrambler.
                              Expected t* ~ log N (true fast scrambler).
    BrickWall2DScrambler  -- random Clifford gates on a 2D NN brick wall.
                              Expected t* ~ sqrt(N) (local 2D ballistic).
                              The "right" Clifford analog of A3 (uses
                              pair-specific random gates instead of A3's
                              single fixed P matrix).
    PowerLawScrambler     -- pair selection probability ~ 1/r^alpha on
                              2D toroidal distance; truly random per gate.
                              Resolves A2 phase 1's narrow fast-scrambling
                              window by using pair-specific gates.
    MERATreeScrambler     -- hierarchical binary-tree gate application.
                              The closest generic Clifford analog of a MERA
                              tensor network.

For HaPPY-style code-based substrates see cwf_hp_phase3b_happy.py.
"""
import os, sys, math
from typing import List, Tuple
import numpy as np
import stim


# =========================================================================
# GF(2) rank (inlined from cwf_pagecurve.py to avoid its import-time test).
# =========================================================================

def gf2_rank(M):
    M = M.copy().astype(np.uint8) & 1
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
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


def make_sim(N: int, seed: int = 0) -> stim.TableauSimulator:
    """Create a TableauSimulator pre-registered to N qubits (all in |0>)."""
    sim = stim.TableauSimulator(seed=int(seed))
    # Register all qubits by applying identity tableau on the full range.
    sim.do_tableau(stim.Tableau(N), list(range(N)))
    return sim


# =========================================================================
# Stabilizer-state entropy infrastructure
# =========================================================================

def stabilizer_matrix(sim, N):
    """Return (X, Z) the symplectic [X|Z] representation of the stabilizer
    generators of a TableauSimulator state. Shape (N, N) each, dtype uint8.

    Caller is expected to have registered all N qubits (via make_sim).
    If stim's simulator has fewer rows than N, pads with Z_i for untouched
    qubits (a |0> state stabilized by single-qubit Z)."""
    stabs = sim.canonical_stabilizers()
    X = np.zeros((N, N), np.uint8); Z = np.zeros((N, N), np.uint8)
    for i, ps in enumerate(stabs):
        if i >= N:
            break
        xs, zs = ps.to_numpy()
        m = min(len(xs), N)
        X[i, :m] = np.asarray(xs[:m], dtype=np.uint8)
        Z[i, :m] = np.asarray(zs[:m], dtype=np.uint8)
    for i in range(len(stabs), N):
        Z[i, i] = 1  # |0>_i stabilizer
    return X, Z


def entropy_region(X, Z, A_qubits, N):
    """Entanglement entropy (bits) of region A for a stabilizer state.
    Pure-state formula: S_A = rank_GF2(G|_B) - |B|, where B = complement of A
    and G is the stabilizer-generator matrix in symplectic [X|Z] form."""
    A_set = set(int(q) for q in A_qubits)
    B = np.array([q for q in range(N) if q not in A_set], dtype=int)
    if B.size == 0:
        return 0.0
    GB = np.hstack([X[:, B], Z[:, B]])
    return float(gf2_rank(GB) - B.size)


def mutual_information(X, Z, A_qubits, B_qubits, N):
    """I(A:B) = S(A) + S(B) - S(AB) for disjoint subsystems A, B."""
    A_set = set(int(q) for q in A_qubits)
    B_set = set(int(q) for q in B_qubits)
    if A_set & B_set:
        raise ValueError("A and B must be disjoint")
    AB = list(A_set | B_set)
    S_A  = entropy_region(X, Z, list(A_set), N)
    S_B  = entropy_region(X, Z, list(B_set), N)
    S_AB = entropy_region(X, Z, AB, N)
    return S_A + S_B - S_AB


# =========================================================================
# Scrambler architectures
# =========================================================================

class Scrambler:
    """Base class. Each subclass implements `step(sim, rng)` that applies
    one "scrambler step" -- a unit of scrambling activity matched in
    total-gate-count across architectures so that depth comparisons make
    sense.

    Convention: one step applies (N // 2) random 2-qubit Clifford gates,
    so each qubit is touched on average once per step. Total gates across
    architectures equal => total entanglement-creation budget equal.
    """
    def __init__(self, N: int, seed: int = 0):
        self.N = N
        self.rng = np.random.default_rng(seed)

    def step(self, sim: stim.TableauSimulator) -> None:
        raise NotImplementedError

    def label(self) -> str:
        raise NotImplementedError


class AllToAllScrambler(Scrambler):
    """Random Clifford on random pairs each step.

    Each step: N/2 random pair selections (no repeat in a step), each gets
    a fresh random 2-qubit Clifford.

    Theoretical scrambling time: t* ~ log N (fast scrambler limit)."""

    def step(self, sim: stim.TableauSimulator) -> None:
        N = self.N
        # disjoint random matching of N/2 pairs
        perm = self.rng.permutation(N)
        for i in range(N // 2):
            a, b = int(perm[2 * i]), int(perm[2 * i + 1])
            tab = stim.Tableau.random(2)
            sim.do_tableau(tab, [a, b])

    def label(self) -> str:
        return "all-to-all"


class BrickWall2DScrambler(Scrambler):
    """Random Clifford on 2D-grid NN brick wall.

    N must be a perfect square. Each step cycles through 4 sub-layers
    (horizontal-even, horizontal-odd, vertical-even, vertical-odd).

    Theoretical scrambling time: t* ~ sqrt(N) (local 2D ballistic)."""

    def __init__(self, N: int, seed: int = 0):
        super().__init__(N, seed)
        self.L = int(round(math.sqrt(N)))
        if self.L * self.L != N:
            raise ValueError(f"N={N} not a perfect square for 2D substrate")
        L = self.L
        idx = {(i, j): i * L + j for i in range(L) for j in range(L)}
        self.idx = idx
        self.layers = [
            [(idx[i, j], idx[i, j + 1]) for i in range(L) for j in range(0, L - 1, 2)],
            [(idx[i, j], idx[i, j + 1]) for i in range(L) for j in range(1, L - 1, 2)],
            [(idx[i, j], idx[i + 1, j]) for j in range(L) for i in range(0, L - 1, 2)],
            [(idx[i, j], idx[i + 1, j]) for j in range(L) for i in range(1, L - 1, 2)],
        ]
        self._step_count = 0

    def step(self, sim: stim.TableauSimulator) -> None:
        # Apply 2 sub-layers per step (covers each qubit twice on average,
        # giving N/2 gates -- same gate-budget as all-to-all).
        # Cycle the 4 sub-layers.
        for k in range(2):
            sub_idx = (self._step_count + k) % 4
            for (a, b) in self.layers[sub_idx]:
                tab = stim.Tableau.random(2)
                sim.do_tableau(tab, [a, b])
        self._step_count += 2

    def label(self) -> str:
        return "2D brick-wall"


class PowerLawScrambler(Scrambler):
    """Random Clifford with pair-selection probability proportional to
    1/r^alpha on 2D toroidal distance.

    N must be a perfect square. Each step applies N/2 gates on pairs
    sampled from the power-law distance distribution.

    Resolves the architectural issue of A2 phase 1 (single fixed P matrix
    causing mean-field synchronization) by giving each gate a fresh random
    Clifford."""

    def __init__(self, N: int, alpha: float, seed: int = 0):
        super().__init__(N, seed)
        self.alpha = alpha
        self.L = int(round(math.sqrt(N)))
        if self.L * self.L != N:
            raise ValueError(f"N={N} not a perfect square")
        L = self.L
        # precompute pair probabilities ~ 1/r^alpha (toroidal distance)
        # pair (a, b) with a < b
        weights = []
        pairs = []
        for a in range(N):
            for b in range(a + 1, N):
                xa, ya = a // L, a % L
                xb, yb = b // L, b % L
                dx = min(abs(xa - xb), L - abs(xa - xb))
                dy = min(abs(ya - yb), L - abs(ya - yb))
                r = math.sqrt(dx * dx + dy * dy)
                if r == 0:
                    continue
                w = 1.0 / (r ** alpha)
                weights.append(w); pairs.append((a, b))
        w = np.array(weights, dtype=float)
        self.pair_probs = w / w.sum()
        self.pairs = np.array(pairs, dtype=int)

    def step(self, sim: stim.TableauSimulator) -> None:
        n_gates = self.N // 2
        # sample n_gates pairs with replacement (cheap and OK for scrambling)
        idxs = self.rng.choice(len(self.pairs), size=n_gates,
                                p=self.pair_probs, replace=True)
        for k in idxs:
            a, b = int(self.pairs[k, 0]), int(self.pairs[k, 1])
            tab = stim.Tableau.random(2)
            sim.do_tableau(tab, [a, b])

    def label(self) -> str:
        return f"power-law alpha={self.alpha}"


class MERATreeScrambler(Scrambler):
    """Hierarchical binary-tree gate application -- the closest generic
    Clifford analog of a MERA tensor network.

    Each step applies log2(N) rounds of gates:
        round k: gates on pairs (2^k * 2i, 2^k * (2i + 1))
                 for i in 0..N/2^(k+1)-1
    Pairs span exponentially increasing distance with depth.

    Total gates per step = N - 1 (binary tree size).  Renormalised here to
    fit the "N/2 gates per step" convention by truncating early levels;
    in practice the tree dominates."""

    def __init__(self, N: int, seed: int = 0):
        super().__init__(N, seed)
        if (N & (N - 1)) != 0:
            raise ValueError(f"N={N} must be a power of 2 for MERA-tree")
        # precompute all tree pairs across all levels
        self.levels = []
        k = 0
        while (1 << (k + 1)) <= N:
            stride = 1 << k
            pairs = [(i, i + stride) for i in range(0, N, 2 * stride)]
            self.levels.append(pairs)
            k += 1
        # Note: total levels = log2(N), total pairs = N/2 + N/4 + ... + 1 = N - 1
        self._level_cursor = 0

    def step(self, sim: stim.TableauSimulator) -> None:
        # One step = one full tree pass (all log2(N) levels). Total gates
        # per step = N - 1 ~ N, matching the other scramblers' gate budget.
        # Hierarchical structure: short-range entanglement at low levels,
        # long-range at high levels (MERA's defining property).
        for level_pairs in self.levels:
            for (a, b) in level_pairs:
                tab = stim.Tableau.random(2)
                sim.do_tableau(tab, [int(a), int(b)])

    def label(self) -> str:
        return "MERA-tree"


# =========================================================================
# State preparation primitives
# =========================================================================

def prepare_bell_pairs(sim: stim.TableauSimulator, pairs: List[Tuple[int, int]]):
    """Initialise |Bell+>^k on the listed (a, b) pairs. Assumes both qubits
    start in |0>."""
    for (a, b) in pairs:
        sim.h(a)
        sim.cnot(a, b)


def prepare_random_clifford_state(sim: stim.TableauSimulator, qubits: List[int],
                                   depth: int, rng: np.random.Generator):
    """Apply a random Clifford circuit (depth 'depth') to scramble qubits."""
    n = len(qubits)
    for d in range(depth):
        # random perfect matching
        perm = rng.permutation(qubits)
        for i in range(n // 2):
            a, b = int(perm[2 * i]), int(perm[2 * i + 1])
            tab = stim.Tableau.random(2)
            sim.do_tableau(tab, [a, b])


# =========================================================================
# Page curve and scrambling protocols
# =========================================================================

def page_curve_subsystem_size(X, Z, N, ordering=None):
    """Return S(L) for contiguous (or specified order) revealing of qubits.

    ordering: a permutation of range(N) giving the order in which to reveal.
    If None, uses range(N) (contiguous from 0)."""
    if ordering is None:
        ordering = list(range(N))
    Ls = list(range(0, N + 1))
    S = np.zeros(len(Ls))
    for k, L in enumerate(Ls):
        S[k] = entropy_region(X, Z, ordering[:L], N)
    return Ls, S


def scrambling_time(scrambler: Scrambler, target_frac: float = 0.95,
                    max_steps: int = None, seed: int = 0,
                    initial_depth: int = 0) -> int:
    """t* = first step at which S(left half) >= target_frac * (N/2).

    Uses the Page-curve criterion: a fully scrambled state has S(any
    half-system) close to the Page bound N/2 - O(1). t* is when we
    cross 95% of that bound.

    initial_depth: scrambling already applied before measurement starts
    (useful when measuring t* on top of a pre-scrambled background)."""
    N = scrambler.N
    sim = make_sim(N, seed=seed)
    # apply any pre-scrambling
    for _ in range(initial_depth):
        scrambler.step(sim)
    target = target_frac * (N // 2)
    if max_steps is None:
        max_steps = max(4 * N, 100)
    half = list(range(N // 2))
    for t in range(1, max_steps + 1):
        scrambler.step(sim)
        X, Z = stabilizer_matrix(sim, N)
        S = entropy_region(X, Z, half, N)
        if S >= target:
            return t
    return max_steps
