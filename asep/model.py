import numpy as np
from .mc import run_mc_batched, make_uniforms
from .bkl import run_bkl_fenwick


class TwoChannelASEP:
    """
    Two-channel Asymmetric Simple Exclusion Process with narrow entrances.

    Implements the model from Pronina & Kolomeisky (2007, J. Phys. A 40, 2275).

    Two parallel 1D lattices ("channels"), each with L sites (0-indexed).
    - Channel 1: particles hop RIGHT (sites 0 -> L-1), enter at site 0, exit at site L-1
    - Channel 2: particles hop LEFT (sites L-1 -> 0), enter at site L-1, exit at site 0
    - Hard-core exclusion: max 1 particle per site

    Narrow entrance coupling:
    - Particle enters channel 1 at site 0 (rate alpha) ONLY if lane2[0] (exit of ch2) is empty
    - Particle enters channel 2 at site L-1 (rate alpha) ONLY if lane1[L-1] (exit of ch1) is empty
    - Exits (rate beta) are independent of the other channel

    Parameters
    ----------
    L : int
        Lattice size (number of sites per channel)
    alpha : float
        Entrance rate (0, 1]
    beta : float
        Exit rate (0, 1]
    seed : int, optional
        Random seed for reproducibility
    """

    def __init__(self, L: int, alpha: float, beta: float, seed: int = 42):
        self.L = L
        self.alpha = alpha
        self.beta = beta
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        self.lane1 = np.zeros(L, dtype=np.int8)
        self.lane2 = np.zeros(L, dtype=np.int8)

        # Counters for observables
        self.total_time = 0.0
        self.current1 = 0  # number of particles exited from channel 1
        self.current2 = 0  # number of particles exited from channel 2

        # Sampling
        self._density_samples1 = []
        self._density_samples2 = []
        self._time_samples = []
        self._site_density1 = np.zeros(L, dtype=np.float64)
        self._site_density2 = np.zeros(L, dtype=np.float64)
        self._n_samples = 0

        # Joint density samples (rho1, rho2) for SSB detection
        self._joint_samples = []

    def run(self, n_steps: int, sample_every: int = 100, warmup: int = 0,
            use_bkl: bool = True):
        """
        Run the simulation for `n_steps` Monte Carlo steps.

        Parameters
        ----------
        n_steps : int
            Number of Monte Carlo steps to execute
        sample_every : int
            Sample density profiles every N steps
        warmup : int
            Number of initial steps to discard before sampling
        use_bkl : bool
            Use the BKL Fenwick-tree kernel (~4x faster, same physics) instead
            of the full-scan Gillespie kernel.
        """
        block = 1000
        done = 0
        while done < n_steps:
            step = min(block, n_steps - done)
            uniforms = make_uniforms(step * 3, self._rng)
            if use_bkl:
                dt, e1, e2, _ = run_bkl_fenwick(
                    self.lane1, self.lane2, self.alpha, self.beta, step,
                    uniforms, 0
                )
            else:
                dt, e1, e2 = run_mc_batched(
                    self.lane1, self.lane2, self.alpha, self.beta, step,
                    uniforms
                )
            self.total_time += dt
            self.current1 += e1
            self.current2 += e2
            done += step
            if done > warmup and done % sample_every < block:
                self._density_samples1.append(np.mean(self.lane1))
                self._density_samples2.append(np.mean(self.lane2))
                self._time_samples.append(self.total_time)
                self._site_density1 += self.lane1.astype(np.float64)
                self._site_density2 += self.lane2.astype(np.float64)
                self._n_samples += 1
                self._joint_samples.append((np.mean(self.lane1), np.mean(self.lane2)))

    def get_bulk_densities(self):
        """
        Return bulk densities (mean over all sites, time-averaged).
        """
        if self._n_samples > 0:
            rho1 = np.mean(self._density_samples1)
            rho2 = np.mean(self._density_samples2)
        else:
            rho1 = np.mean(self.lane1)
            rho2 = np.mean(self.lane2)
        return rho1, rho2

    def get_currents(self):
        """
        Return steady-state particle currents (particles exited per unit time).
        """
        if self.total_time > 0:
            J1 = self.current1 / self.total_time
            J2 = self.current2 / self.total_time
        else:
            J1, J2 = 0.0, 0.0
        return J1, J2

    def get_site_density_profiles(self):
        """
        Return time-averaged density profile for each channel.
        """
        if self._n_samples == 0:
            return self.lane1.astype(float), self.lane2.astype(float)
        return self._site_density1 / self._n_samples, self._site_density2 / self._n_samples

    def get_joint_density_samples(self):
        """
        Return the list of simultaneous (rho1, rho2) bulk-density samples.

        Used to build the joint density distribution P(rho1, rho2) for
        spontaneous-symmetry-breaking detection (bimodal in SSB phases).
        """
        return np.array(self._joint_samples)

    def reset(self):
        """Reset simulation to empty lattice and reseed the RNG."""
        self.lane1[:] = 0
        self.lane2[:] = 0
        self._rng = np.random.default_rng(self.seed)
        self.total_time = 0.0
        self.current1 = 0
        self.current2 = 0
        self._density_samples1 = []
        self._density_samples2 = []
        self._time_samples = []
        self._site_density1 = np.zeros(self.L, dtype=np.float64)
        self._site_density2 = np.zeros(self.L, dtype=np.float64)
        self._n_samples = 0
        self._joint_samples = []
