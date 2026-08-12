"""
Phase classification utilities for the two-channel ASEP.

Classification based on measured currents J1, J2 and bulk densities rho1, rho2,
compared against the mean-field predictions of Pronina & Kolomeisky (2007).
"""
import numpy as np


def classify_phase(J1, J2, rho1, rho2, alpha, beta, L,
                   asym_threshold=0.1, mc_tol=0.05):
    """
    Classify the stationary phase of the two-channel ASEP.

    The system has a Z2 symmetry (channel 1 <-> channel 2). Phases
    (Pronina & Kolomeisky 2007, Sec 2.2):

    1. LD   - symmetric, both channels low density
    2. MC   - symmetric, maximal current (rho ~ 1/2, J ~ 1/4)
    3. HD/LD- asymmetric: one channel high density (rho ~ 1-beta), other low
    4. LD/LD- asymmetric: both low density but different

    Returns
    -------
    label : str
        'LD', 'MC', 'HD/LD', 'LD/HD', or 'LD/LD'
    asymmetry : float
        |rho1 - rho2| (measure of symmetry breaking)
    """
    delta_rho = abs(rho1 - rho2)
    delta_J = abs(J1 - J2)
    asymmetry = max(delta_rho, delta_J)

    if asymmetry < asym_threshold:
        # Symmetric phase
        J_avg = (J1 + J2) / 2
        rho_avg = (rho1 + rho2) / 2
        if abs(J_avg - 0.25) < mc_tol and abs(rho_avg - 0.5) < mc_tol:
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
