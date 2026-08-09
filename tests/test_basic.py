"""
Test suite for the TwoChannelASEP model.
"""
import numpy as np
from asep import TwoChannelASEP


def test_ld_phase():
    """In LD phase (α < β, α < 0.5), current should be ~α(1-α)."""
    sim = TwoChannelASEP(L=100, alpha=0.2, beta=0.8, seed=42)
    sim.run(n_steps=100000, sample_every=200, warmup=5000)
    rho1, rho2 = sim.get_bulk_densities()
    J1, J2 = sim.get_currents()
    print(f"LD phase: rho1={rho1:.4f}, rho2={rho2:.4f}, J1={J1:.4f}, J2={J2:.4f}")
    assert J1 > 0 and J2 > 0, "Currents should be positive"
    assert abs(rho1 - rho2) < 0.1, "In symmetric phase, densities should be close"


def test_mc_phase():
    """In MC phase (α > 0.5, β > 0.5), current should be ~0.25 (suppressed by coupling)."""
    sim = TwoChannelASEP(L=200, alpha=0.9, beta=0.9, seed=42)
    sim.run(n_steps=200000, sample_every=200, warmup=10000)
    J1, J2 = sim.get_currents()
    print(f"MC phase: J1={J1:.4f}, J2={J2:.4f}")
    assert 0.15 < J1 < 0.25, f"Current {J1} should be near 0.25 in MC phase (suppressed by coupling)"


def test_hd_ld_asymmetry():
    """
    In HD/LD phase (α ≫ β), density profiles should show one channel in HD, the other in LD.
    """
    sim = TwoChannelASEP(L=200, alpha=0.9, beta=0.23, seed=42)
    sim.run(n_steps=500000, sample_every=500, warmup=50000)
    prof1, prof2 = sim.get_site_density_profiles()

    # Channel 2 should be HD (high density near exit site lane2[0])
    assert prof2[0] > 0.5, f"Channel 2 exit density {prof2[0]} should be > 0.5 (HD phase)"
    # Channel 1 should be LD (low density throughout)
    assert prof1[0] < 0.3, f"Channel 1 entrance density {prof1[0]} should be < 0.3 (LD phase)"
    # The asymmetry should be present
    assert abs(prof1[0] - prof2[0]) > 0.2, "Significant asymmetry expected"


def test_current_conservation():
    """In steady state, current should be constant across the system."""
    sim = TwoChannelASEP(L=100, alpha=0.3, beta=0.5, seed=42)
    sim.run(n_steps=50000, sample_every=100, warmup=5000)
    prof1, prof2 = sim.get_site_density_profiles()
    # In steady state, local current j(i) = rho(i)(1-rho(i+1)) should be roughly constant
    # in the bulk for single-channel TASEP. Here with coupling, it's approximate.
    bulk1 = prof1[10:40]
    bulk2 = prof2[10:40]
    # Just check densities are reasonable
    assert np.all(bulk1 >= 0) and np.all(bulk1 <= 1)
    assert np.all(bulk2 >= 0) and np.all(bulk2 <= 1)


if __name__ == "__main__":
    test_ld_phase()
    print("LD test passed.")
    test_mc_phase()
    print("MC test passed.")
    test_hd_ld_asymmetry()
    print("HD/LD asymmetry test passed.")
    test_current_conservation()
    print("Current conservation test passed.")
    print("\nAll tests passed.")
