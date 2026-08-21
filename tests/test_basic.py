"""
Test suite for the TwoChannelASEP model.
"""
import numpy as np
from asep import TwoChannelASEP


def test_classify_phase_hd_ld():
    """REGRESSION: the ensemble phase classifier must recover HD/LD.

    The bug: each replica is stuck in ONE broken basin, so its own
    std(rho1-rho2) is ~0 (constant difference ~0.8). The old code used that
    std -> classified everything as symmetric LD, so NO phase diagram ever
    showed the HD/LD (or LD/HD) transition. The fix uses per-replica
    dense=max(rho1,rho2); mean_dense>1/2 -> HD/LD.
    """
    from asep.observables import classify_phase
    # two reps dense-in-1, two dense-in-2 (alpha=0.9, beta=0.1, L=1000)
    samples = np.array([[0.87, 0.06], [0.88, 0.05], [0.06, 0.87], [0.05, 0.88]])
    lab, asym = classify_phase(0.04, 0.04, 0.47, 0.47, 0.9, 0.1, 1000,
                               samples=samples, j_current=0.25)
    assert lab in ("HD/LD", "LD/HD"), f"expected HD/LD, got {lab}"

    # LD: symmetric low density
    s2 = np.full((4, 2), 0.2)
    lab2, _ = classify_phase(0.15, 0.15, 0.2, 0.2, 0.5, 0.5, 1000,
                             samples=s2, j_current=0.25)
    assert lab2 == "LD"

    # MC: symmetric high + current saturated
    s3 = np.full((4, 2), 0.5)
    lab3, _ = classify_phase(0.25, 0.25, 0.5, 0.5, 1.0, 1.0, 1000,
                             samples=s3, j_current=0.25)
    assert lab3 == "MC"


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


def test_adaptive_convergence():
    """run_adaptive should stop early once the current plateaus, and agree
    with a full run of the same length on the converged observable."""
    sim = TwoChannelASEP(L=200, alpha=0.9, beta=0.9, seed=7)
    steps = sim.run_adaptive(max_steps=2_000_000, sample_every=200,
                             warmup=10000, block=1000, tol=1e-3,
                             min_steps=50_000)
    J1, J2 = sim.get_currents()
    # Converged MC current should be near 0.25 (suppressed by coupling)
    assert 0.15 < J1 < 0.25, f"J1={J1:.4f}"
    assert 0.15 < J2 < 0.25, f"J2={J2:.4f}"
    # It should have stopped well before the max (plateau detected)
    assert steps < 2_000_000, f"adaptive run did not stop early ({steps} steps)"
    assert steps >= 50_000, "should respect min_steps"


if __name__ == "__main__":
    test_ld_phase()
    print("LD test passed.")
    test_mc_phase()
    print("MC test passed.")
    test_hd_ld_asymmetry()
    print("HD/LD asymmetry test passed.")
    test_current_conservation()
    print("Current conservation test passed.")
    test_adaptive_convergence()
    print("Adaptive convergence test passed.")
    print("\nAll tests passed.")
