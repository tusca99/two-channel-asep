import numpy as np


class TwoChannelASEP:
    """
    Two-channel Asymmetric Simple Exclusion Process with narrow entrances.

    Implements the model from Pronina & Kolomeisky (2007, J. Phys. A 40, 2275).

    Two parallel 1D lattices ("channels"), each with L sites (0-indexed).
    - Channel 1: particles hop RIGHT (sites 0 -> L-1), enter at site 0, exit at site L-1
    - Channel 2: particles hop LEFT (sites L-1 -> 0), enter at site L-1, exit at site 0
    - Hard-core exclusion: max 1 particle per site

    Narrow entrance coupling:
    - Particle enters channel 1 at site 0 (rate alpha) ONLY if lane2[L-1] is empty
    - Particle enters channel 2 at site L-1 (rate alpha) ONLY if lane1[0] is empty
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
        self.rng = np.random.default_rng(seed)

        self.lane1 = np.zeros(L, dtype=np.int8)
        self.lane2 = np.zeros(L, dtype=np.int8)

        # Counters for observables
        self.total_time = 0.0
        self.current1 = 0  # number of particles exited from channel 1
        self.current2 = 0  # number of particles exited from channel 2
        self.n_attempts = 0

        # Sampling
        self._density_samples1 = []
        self._density_samples2 = []
        self._time_samples = []
        self._site_density1 = np.zeros(L, dtype=np.float64)
        self._site_density2 = np.zeros(L, dtype=np.float64)
        self._n_samples = 0

    def step(self):
        """
        One Gillespie (continuous-time) Monte Carlo step.

        Collects all possible moves with their rates, computes total rate,
        picks a move proportional to its rate, and advances time by dt ~ Exp(total_rate).
        """
        L = self.L
        lane1 = self.lane1
        lane2 = self.lane2
        alpha = self.alpha
        beta = self.beta

        # Collect all possible moves as (description, site, rate)
        moves = []

        # Bulk hopping: channel 1 right (site i -> i+1), channel 2 left (site i -> i-1)
        for i in range(L - 1):
            if lane1[i] == 1 and lane1[i + 1] == 0:
                moves.append(('hop1_right', i, 1.0))
            if lane2[i + 1] == 1 and lane2[i] == 0:
                moves.append(('hop2_left', i + 1, 1.0))

        # Entrance events
        # Enter channel 1 at site 0: requires lane1[0]=0 AND lane2[0] (exit site of ch2)=0
        if lane1[0] == 0 and lane2[0] == 0:
            moves.append(('enter1', 0, alpha))
        # Enter channel 2 at site L-1: requires lane2[L-1]=0 AND lane1[L-1] (exit site of ch1)=0
        if lane2[L - 1] == 0 and lane1[L - 1] == 0:
            moves.append(('enter2', L - 1, alpha))

        # Exit events (independent of other channel)
        if lane1[L - 1] == 1:
            moves.append(('exit1', L - 1, beta))
        if lane2[0] == 1:
            moves.append(('exit2', 0, beta))

        if not moves:
            # System fully blocked - advance time with a minimal step
            self.total_time += 0.001
            return

        total_rate = sum(m[2] for m in moves)
        dt = self.rng.exponential(1.0 / total_rate)
        self.total_time += dt

        # Select one move proportional to its rate
        r = self.rng.random() * total_rate
        cum = 0.0
        for desc, site, rate in moves:
            cum += rate
            if r <= cum:
                self._apply_move(desc, site)
                break

        self.n_attempts += 1

    def _apply_move(self, desc: str, site: int):
        """Execute a single move on the lattices."""
        L = self.L
        lane1 = self.lane1
        lane2 = self.lane2

        if desc == 'hop1_right':
            lane1[site] = 0
            lane1[site + 1] = 1
        elif desc == 'hop2_left':
            lane2[site] = 0
            lane2[site - 1] = 1
        elif desc == 'enter1':
            lane1[0] = 1
        elif desc == 'enter2':
            lane2[L - 1] = 1
        elif desc == 'exit1':
            lane1[L - 1] = 0
            self.current1 += 1
        elif desc == 'exit2':
            lane2[0] = 0
            self.current2 += 1

    def run(self, n_steps: int, sample_every: int = 100, warmup: int = 0):
        """
        Run the simulation for `n_steps` Gillespie steps.

        Parameters
        ----------
        n_steps : int
            Number of Monte Carlo steps to execute
        sample_every : int
            Sample density profiles every N steps
        warmup : int
            Number of initial steps to discard before sampling
        """
        for i in range(n_steps):
            self.step()
            if i > warmup and i % sample_every == 0:
                self._density_samples1.append(np.mean(self.lane1))
                self._density_samples2.append(np.mean(self.lane2))
                self._time_samples.append(self.total_time)
                self._site_density1 += self.lane1.astype(np.float64)
                self._site_density2 += self.lane2.astype(np.float64)
                self._n_samples += 1

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

    def reset(self):
        """Reset simulation to empty lattice."""
        self.lane1[:] = 0
        self.lane2[:] = 0
        self.total_time = 0.0
        self.current1 = 0
        self.current2 = 0
        self.n_attempts = 0
        self._density_samples1 = []
        self._density_samples2 = []
        self._time_samples = []
        self._site_density1 = np.zeros(self.L, dtype=np.float64)
        self._site_density2 = np.zeros(self.L, dtype=np.float64)
        self._n_samples = 0
