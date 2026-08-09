"""
Phase classification utilities for the two-channel ASEP.

Classification based on measured currents J1, J2 and bulk densities rho1, rho2.
"""
import numpy as np


def classify_phase(J1, J2, rho1, rho2, alpha, beta, L, asymmetry_threshold=0.03):
    """
    Classify the stationary phase of the two-channel ASEP.

    The system has a Z2 symmetry (channel 1 ↔ channel 2).
    Phases (from Pronina & Kolomeisky 2007, §2.2):

    1. LD  - symmetric, both channels low density
    2. MC  - symmetric, maximal current
    3. HD/LD - asymmetric: one channel high density, other low density
    4. LD/LD - asymmetric: both low density but different

    Parameters
    ----------
    J1, J2 : float
        Steady-state currents in channels 1 and 2
    rho1, rho2 : float
        Bulk densities in channels 1 and 2
    alpha, beta : float
        Entrance and exit rates
    asymmetry_threshold : float
        Minimum |J1-J2| or |rho1-rho2| to classify as asymmetric

    Returns
    -------
    label : str
        Phase label: 'LD', 'MC', 'HD/LD', 'LD/LD', or 'LD/HD'
    asymmetry : float
        |rho1 - rho2| (measure of symmetry breaking)
    """
    delta_J = abs(J1 - J2)
    delta_rho = abs(rho1 - rho2)
    asymmetry = max(delta_J, delta_rho)

    is_asymmetric = asymmetry > asymmetry_threshold

    if not is_asymmetric:
        # Symmetric phase
        J_avg = (J1 + J2) / 2
        rho_avg = (rho1 + rho2) / 2

        if J_avg > 0.18 and alpha > 0.5 and beta > 0.5:
            # Near MC regime
            label = "MC"
        elif alpha < 0.5 and beta > alpha:
            label = "LD"
        else:
            # Could also be HD-like, but in this model with narrow entrances,
            # the symmetric HD phase doesn't exist (as noted in paper §2.2)
            label = "HD"
    else:
        # Asymmetric phase
        # Determine which channel has higher density
        rho_hd_theory = 1 - beta

        if rho1 > rho2:
            # Channel 1 has higher density
            if rho1 > 0.35 and rho_hd_theory > 0.3:
                label = "HD/LD"  # ch1 HD, ch2 LD
            else:
                label = "LD/LD"
        else:
            # Channel 2 has higher density
            if rho2 > 0.35 and rho_hd_theory > 0.3:
                label = "LD/HD"  # ch2 HD, ch1 LD
            else:
                label = "LD/LD"

    return label, asymmetry


def classify_phase_simple(rho1, rho2, J1, J2, alpha, beta, L):
    """
    Simple classification for quick analysis.
    Returns a short phase label.
    """
    delta_rho = abs(rho1 - rho2)

    if delta_rho < 0.05:
        # Symmetric
        if (J1 + J2) / 2 > 0.18:
            return "MC"
        return "LD"
    else:
        # Asymmetric - which channel is higher?
        rho_hd_pred = 1 - beta
        if rho_hd_pred > 0.4:
            if rho1 > rho2:
                return "HD/LD"
            else:
                return "LD/HD"
        return "LD/LD"
