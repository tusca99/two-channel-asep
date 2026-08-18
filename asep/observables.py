"""
Phase classification utilities for the two-channel ASEP.

Classification based on the joint density distribution P(rho1, rho2),
following the density-distribution method of Pronina & Kolomeisky (2007).

At large L the system can flip between the two broken-symmetry states, so the
time-averaged |rho1 - rho2| washes out. The robust SSB order parameter is the
standard deviation of (rho1 - rho2): large when the system visits broken
states (bimodal P), small when it is symmetric (unimodal P on the diagonal).
"""
import numpy as np


def classify_phase(J1, J2, rho1, rho2, alpha, beta, L,
                   asym_threshold=0.05, mc_rho=0.45, samples=None,
                   j_current=None):
    """
    Classify the stationary phase of the two-channel ASEP.

    Symmetry breaking is detected via std(rho1 - rho2) (robust to the
    state-flipping that washes out the time-averaged |rho1-rho2| at large L).

    MC detection: a fixed density threshold (mc_rho) misplaces the LD/MC
    boundary when density rises slowly toward 1/2. The paper (Fig 4) locates
    the LD/MC boundary by *current saturation*: J saturates at 1/4 in MC.
    So we classify as MC when the current is within `mc_tol` of 1/4 AND the
    density is high, which is the robust signature.

    Parameters
    ----------
    J1, J2 : float
        Steady-state currents in channels 1 and 2
    rho1, rho2 : float
        Time-averaged bulk densities in channels 1 and 2
    alpha, beta : float
        Entrance and exit rates
    L : int
        Lattice size
    samples : (N,2) array, optional
        Joint (rho1, rho2) density samples for the density-distribution method
    j_current : float, optional
        Saturated/current-saturation value. If given and J1+J2 is within
        mc_tol of 2*j_current, MC is identified by current saturation first.

    Returns
    -------
    label : str
        'LD', 'MC', 'HD/LD', 'LD/HD', or 'LD/LD'
    asymmetry : float
        std(rho1 - rho2) if samples given, else |rho1 - rho2|
    """
    if samples is not None and len(samples) > 0:
        d = samples[:, 0] - samples[:, 1]
        asymmetry = d.std()
    else:
        asymmetry = abs(rho1 - rho2)

    rho_avg = (rho1 + rho2) / 2

    # MC by current saturation (paper Fig 4 method): J plateaus at 1/4 in MC.
    mc_tol = 0.02
    j_cur = j_current if j_current is not None else 0.25
    j_sat = (J1 + J2) / 2 >= j_cur - mc_tol

    if asymmetry < asym_threshold:
        # Symmetric phase
        if j_sat and rho_avg > mc_rho:
            return "MC", asymmetry
        return "LD", asymmetry

    # Asymmetric phase: which channel is denser?
    # HD/LD: the dense channel has rho ~ 1-beta (> 1/2); LD/LD: both low.
    rho_hd_theory = 1 - beta
    if rho1 > rho2:
        if rho1 > 0.5 and abs(rho1 - rho_hd_theory) < 0.25:
            return "HD/LD", asymmetry
        return "LD/LD", asymmetry
    else:
        if rho2 > 0.5 and abs(rho2 - rho_hd_theory) < 0.25:
            return "LD/HD", asymmetry
        return "LD/LD", asymmetry


def classify_phase_simple(rho1, rho2, J1, J2, alpha, beta, L):
    """Short phase label for quick analysis."""
    label, _ = classify_phase(J1, J2, rho1, rho2, alpha, beta, L)
    return label
