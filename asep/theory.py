"""
Mean-field theory predictions for the two-channel ASEP (Pronina & Kolomeisky 2007).

Provides per-channel MFT values for currents (J1, J2) and bulk densities
(rho1, rho2) in each phase, for overlaying theoretical lines on MC plots.

Phase conditions and values (paper Sec 2.2):
- MC  : J1 = J2 = 1/4,  rho1 = rho2 = 1/2,  for beta>1/2, alpha > 2beta/(4beta-1)
- LD  : J1 = J2 = a1(1-a1), a1 = [alpha+beta - sqrt((alpha+beta)^2 - 4 alpha^2 beta)]/(2 alpha)
- HD/LD: J1 = beta(1-beta), rho1 = 1-beta (channel 1 dense)
         J2 = a2(1-a2), rho2 = a2, a2 = alpha*beta
  (or the Z2 mirror: channel 2 dense)
- LD/LD: both channels low but different densities (small region)
"""
import numpy as np


def alpha1_ld(alpha, beta):
    """Effective entrance rate in the LD phase (eq 12)."""
    disc = (alpha + beta) ** 2 - 4 * alpha**2 * beta
    if disc < 0:
        return np.nan
    return (alpha + beta - np.sqrt(disc)) / (2 * alpha)


def mc_region(alpha, beta):
    """True if (alpha,beta) is in the MC phase (eq 10)."""
    return beta > 0.5 and alpha > 2 * beta / (4 * beta - 1)


def hdld_region(alpha, beta):
    """True if in HD/LD (eq 23)."""
    return beta < alpha / (1 + alpha + alpha**2)


def mft_currents(alpha, beta):
    """
    MFT currents (J1, J2) at a given (alpha, beta).
    """
    if mc_region(alpha, beta):
        return 0.25, 0.25
    if hdld_region(alpha, beta):
        # channel 1 HD, channel 2 LD
        a2 = alpha * beta
        return beta * (1 - beta), a2 * (1 - a2)
    # LD phase
    a1 = alpha1_ld(alpha, beta)
    if np.isnan(a1):
        return np.nan, np.nan
    J = a1 * (1 - a1)
    return J, J


def mft_densities(alpha, beta):
    """
    MFT bulk densities (rho1, rho2) at a given (alpha, beta).
    """
    if mc_region(alpha, beta):
        return 0.5, 0.5
    if hdld_region(alpha, beta):
        # channel 1 HD, channel 2 LD
        return 1 - beta, alpha * beta
    # LD phase
    a1 = alpha1_ld(alpha, beta)
    if np.isnan(a1):
        return np.nan, np.nan
    return a1, a1
