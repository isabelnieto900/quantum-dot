"""
pytest suite for quantum-dot physics helpers.

Validates the finite-well solver against the infinite spherical well and checks
that the optical gap approaches the bulk value at large radius.
"""

from __future__ import annotations

import numpy as np
import pytest

from materials import get_material
from physics import (
    J_to_eV,
    brus_gap,
    confinement_regime,
    effective_mass_kg,
    eV_to_J,
    emission_wavelength_nm,
    finite_spherical_well_energies,
    infinite_well_energy,
    nm_to_m,
    quantum_dot_gap,
    relative_error_vs_infinite,
    wavelength_to_rgb,
)


@pytest.fixture
def cdse():
    return get_material("CdSe")


def test_infinite_well_analytic_scaling():
    """E ∝ 1/R² and ∝ 1/m for the infinite spherical well (l = 0)."""
    m = effective_mass_kg(0.13)
    E1 = infinite_well_energy(nm_to_m(2.0), m, n=1)
    E2 = infinite_well_energy(nm_to_m(4.0), m, n=1)
    assert E1 / E2 == pytest.approx(4.0, rel=1e-10)


def test_hard_wall_matches_infinite_analytic(cdse):
    """Hard-wall discretisation (r_max = R) matches the infinite-well formula."""
    R_m = float(nm_to_m(3.0))
    me = effective_mass_kg(cdse["m_e"])
    err = relative_error_vs_infinite(
        R_m, me, n_points=600, l=0, hard_wall=True
    )
    assert err < 0.01  # < 1 %


def test_deep_finite_well_approaches_infinite(cdse):
    """A very deep finite well approaches the infinite-well energy from below."""
    R_m = float(nm_to_m(3.0))
    me = effective_mass_kg(cdse["m_e"])
    err = relative_error_vs_infinite(
        R_m, me, V0_eV=80.0, n_points=700, l=0, hard_wall=False
    )
    assert err < 0.05  # < 5 % (residual tunnelling expected)


def test_finite_well_bound_below_V0(cdse):
    """Ground-state energy must lie below the continuum threshold V0."""
    R_m = float(nm_to_m(2.5))
    me = effective_mass_kg(cdse["m_e"])
    V0_eV = 2.0
    energies, _, _ = finite_spherical_well_energies(
        R_m, me, float(eV_to_J(V0_eV)), n_points=400, l=0, n_states=1
    )
    assert float(J_to_eV(energies[0])) < V0_eV


def test_gap_tends_to_bulk_at_large_R(cdse):
    """Optical gap should approach Eg when confinement becomes weak."""
    R_large = float(nm_to_m(40.0))
    result = quantum_dot_gap(R_large, cdse, V0_eV=2.0, n_points=400, l=0)
    # At large R both kinetic terms and Coulomb → 0, so gap ≈ Eg.
    assert result["gap_eV"] == pytest.approx(cdse["Eg"], abs=0.08)


def test_brus_decreases_with_radius(cdse):
    """Brus gap falls toward Eg as radius grows (in the usual QD range)."""
    E_small = brus_gap(float(nm_to_m(1.5)), cdse)
    E_large = brus_gap(float(nm_to_m(8.0)), cdse)
    assert E_small > E_large
    assert E_large > cdse["Eg"]


def test_emission_wavelength_and_rgb():
    """λ = hc/E and wavelength_to_rgb returns channels in [0, 1]."""
    lam = emission_wavelength_nm(2.0)
    # 1239.8 / 2 ≈ 619.9 nm
    assert lam == pytest.approx(619.9, rel=1e-3)
    r, g, b = wavelength_to_rgb(lam)
    assert 0.0 <= r <= 1.0 and 0.0 <= g <= 1.0 and 0.0 <= b <= 1.0
    # Outside visible → near black / dark grey.
    dark = wavelength_to_rgb(200.0)
    assert max(dark) < 0.2


def test_confinement_regime_labels(cdse):
    assert confinement_regime(1.0, cdse["a_B"]) == "fuerte"
    assert confinement_regime(cdse["a_B"], cdse["a_B"]) == "intermedio"
    assert confinement_regime(20.0, cdse["a_B"]) == "débil"


def test_l1_solver_runs(cdse):
    """l = 1 should produce a higher ground energy than l = 0 (centrifugal barrier)."""
    R_m = float(nm_to_m(3.0))
    me = effective_mass_kg(cdse["m_e"])
    V0_J = float(eV_to_J(3.0))
    E0, _, _ = finite_spherical_well_energies(R_m, me, V0_J, n_points=400, l=0, n_states=1)
    E1, _, _ = finite_spherical_well_energies(R_m, me, V0_J, n_points=400, l=1, n_states=1)
    assert float(E1[0]) > float(E0[0])
