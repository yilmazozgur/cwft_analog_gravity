"""
CWF computational-black-hole testbed: core substrate models.

Two substrates:
  (1) ReservoirLattice -- a 1D lattice of locally-coupled reservoir nodes.
      Each site i carries an m-dim internal state h_i. Local recurrent gain
      rho_i (the spectral radius of the site's recurrent block) is the
      *recurrence-inertia knob* I_rec. Inter-site nearest-neighbour coupling
      mediates information propagation; its strength sets a finite
      Lieb-Robinson-like velocity (the substrate's "speed of light").

  (2) rule110 -- elementary cellular automaton Rule 110 (Turing-universal),
      used for an exact perturbation light-cone (a clean "the substrate has a c").

All randomness is seeded for reproducibility. No GPU required.
"""

import numpy as np


class ReservoirLattice:
    def __init__(self, n_sites, m=24, coupling=0.18, rho=1.0, leak=0.0,
                 seed=0):
        """
        n_sites : number of lattice sites
        m       : internal dimension per site
        coupling: inter-site coupling strength c (sets propagation budget)
        rho     : either a scalar (homogeneous) or array of length n_sites
                  giving per-site recurrent spectral radius (the I_rec knob)
        leak    : leak/saturation bias (>=0). Larger -> pushes units toward
                  saturation, lowering local Jacobian gain.
        """
        rng = np.random.default_rng(seed)
        self.n = n_sites
        self.m = m
        self.c = coupling
        self.leak = leak
        if np.isscalar(rho):
            self.rho = np.full(n_sites, float(rho))
        else:
            self.rho = np.asarray(rho, dtype=float)
            assert self.rho.shape[0] == n_sites

        # One fixed random recurrent matrix per site, normalised to spectral
        # radius 1, then scaled by rho_i at runtime. Sites share the same
        # random structure (different scaling) so differences are due to rho,
        # not to disorder.
        W = rng.standard_normal((m, m))
        sr = max(abs(np.linalg.eigvals(W)))
        self.W0 = W / sr  # spectral radius exactly 1

        # Fixed coupling projector (same for all bonds).
        Pc = rng.standard_normal((m, m))
        sp = max(abs(np.linalg.eigvals(Pc)))
        self.P = Pc / sp

    def step(self, H):
        """One synchronous update. H: (n, m) array -> (n, m) array."""
        # neighbour sum with open boundaries
        left = np.zeros_like(H)
        right = np.zeros_like(H)
        left[1:] = H[:-1]
        right[:-1] = H[1:]
        neigh = (left + right) @ self.P.T
        recur = (self.rho[:, None] * (H @ self.W0.T))
        leak = self.leak
        if not np.isscalar(leak):
            leak = np.asarray(leak)[:, None]  # per-site -> column broadcast
        pre = recur + self.c * neigh - leak * H
        return np.tanh(pre)

    def run(self, H0, T):
        H = H0.copy()
        traj = np.empty((T + 1, self.n, self.m))
        traj[0] = H
        for t in range(1, T + 1):
            H = self.step(H)
            traj[t] = H
        return traj


# ----------------------------------------------------------------------
# Rule 110 elementary cellular automaton
# ----------------------------------------------------------------------
_RULE110 = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.uint8)  # LSB..MSB


def rule110_step(row):
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    idx = (left << 2) | (row << 1) | right
    return _RULE110[idx]


def rule110_run(row0, T):
    n = row0.shape[0]
    out = np.empty((T + 1, n), dtype=np.uint8)
    out[0] = row0
    r = row0.copy()
    for t in range(1, T + 1):
        r = rule110_step(r)
        out[t] = r
    return out
