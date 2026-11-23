"""
Quantum-dot physics: Brus model, finite spherical well solver, and utilities.

Internal calculations use SI units. Interface-facing helpers accept/return nm and eV.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import constants
from scipy.linalg import eigh_tridiagonal

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------

HBAR = constants.hbar
M_E = constants.m_e
E_CHARGE = constants.e
EPSILON_0 = constants.epsilon_0
H_PLANCK = constants.h
C_LIGHT = constants.c
EV = constants.eV  # Joules per eV


def nm_to_m(x_nm: float | np.ndarray) -> float | np.ndarray:
    """Convert nanometres to metres."""
    return np.asarray(x_nm, dtype=float) * 1e-9


def m_to_nm(x_m: float | np.ndarray) -> float | np.ndarray:
    """Convert metres to nanometres."""
    return np.asarray(x_m, dtype=float) * 1e9


def J_to_eV(E_J: float | np.ndarray) -> float | np.ndarray:
    """Convert joules to electron-volts."""
    return np.asarray(E_J, dtype=float) / EV


def eV_to_J(E_eV: float | np.ndarray) -> float | np.ndarray:
    """Convert electron-volts to joules."""
    return np.asarray(E_eV, dtype=float) * EV


def effective_mass_kg(m_frac: float) -> float:
    """Convert a mass fraction of m_e to kilograms."""
    return float(m_frac) * M_E


# ---------------------------------------------------------------------------
# Brus model and Coulomb correction
# ---------------------------------------------------------------------------

def coulomb_term(R_m: float, epsilon: float) -> float:
    """
    Coulomb correction in eV: 1.8 e² / (4 π ε ε0 R).

    This is the standard first-order exciton correction used in the Brus model.
    """
    energy_J = 1.8 * E_CHARGE**2 / (4.0 * np.pi * epsilon * EPSILON_0 * R_m)
    return float(J_to_eV(energy_J))


def brus_gap(R_m: float, material: dict[str, Any]) -> float:
    """
    Optical gap from the Brus effective-mass model (eV).

    E(R) = Eg + (ħ² π² / 2 R²) (1/me* + 1/mh*) − 1.8 e² / (4 π ε ε0 R)
    """
    me = effective_mass_kg(material["m_e"])
    mh = effective_mass_kg(material["m_h"])
    kinetic_J = (HBAR**2 * np.pi**2 / (2.0 * R_m**2)) * (1.0 / me + 1.0 / mh)
    return float(material["Eg"] + J_to_eV(kinetic_J) - coulomb_term(R_m, material["epsilon"]))


def infinite_well_energy(R_m: float, m_eff_kg: float, n: int = 1) -> float:
    """
    Analytic infinite spherical well energy for radial quantum number n (l = 0).

    E_n = ħ² (n π)² / (2 m R²)   [joules]
    """
    return float((HBAR**2 * (n * np.pi) ** 2) / (2.0 * m_eff_kg * R_m**2))


# ---------------------------------------------------------------------------
# Finite spherical well — finite differences on u(r) = r R(r)
# ---------------------------------------------------------------------------

def finite_spherical_well_energies(
    R_m: float,
    m_eff_kg: float,
    V0_J: float,
    n_points: int = 400,
    l: int = 0,
    r_max_factor: float = 3.0,
    n_states: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve the radial Schrödinger equation for a finite spherical well.

    Potential: V(r) = 0 for r < R, V(r) = V0 for R ≤ r ≤ r_max.
    Domain: r ∈ (0, r_max], r_max = r_max_factor * R.
    Wavefunction variable: u(r) = r R(r), with u(0) = u(r_max) = 0.

    Uses a second-order finite-difference tridiagonal Hamiltonian and
    ``scipy.linalg.eigh_tridiagonal``.

    Parameters
    ----------
    R_m :
        Well radius in metres.
    m_eff_kg :
        Effective mass in kg.
    V0_J :
        Well depth (band offset) in joules. Must be > 0.
    n_points :
        Number of interior grid points.
    l :
        Orbital angular momentum quantum number.
    r_max_factor :
        Outer domain extent in units of R.
    n_states :
        Number of lowest eigenpairs to return.

    Returns
    -------
    energies_J :
        Bound-ish eigenvalues in joules (lowest ``n_states``).
    wavefunctions :
        Array of shape (n_points, n_states) with normalised u(r).
    r_grid :
        Radial grid in metres (interior points).
    """
    if n_points < 50:
        raise ValueError("n_points must be at least 50 for a stable discretisation")
    if V0_J <= 0:
        raise ValueError("V0_J must be positive (finite well depth)")
    if R_m <= 0:
        raise ValueError("R_m must be positive")

    r_max = r_max_factor * R_m
    # Interior points: exclude r=0 and r=r_max where u=0.
    r = np.linspace(0.0, r_max, n_points + 2)[1:-1]
    dr = r[1] - r[0]

    # Kinetic coefficient for the second-derivative stencil.
    t = HBAR**2 / (2.0 * m_eff_kg * dr**2)

    # Diagonal: 2t + V(r) + centrifugal barrier.
    V = np.where(r < R_m, 0.0, V0_J)
    # Avoid division by zero: r starts above 0 by construction.
    centrifugal = (HBAR**2 * l * (l + 1)) / (2.0 * m_eff_kg * r**2)
    diagonal = 2.0 * t + V + centrifugal
    off_diagonal = np.full(n_points - 1, -t)

    # Request a few extra eigenvalues; filter later if needed.
    k = min(n_states, n_points)
    energies, vectors = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, k - 1),
    )

    # Normalise ∫ |u|² dr = 1 (trapezoid on uniform grid ≈ sum * dr).
    for i in range(vectors.shape[1]):
        norm = np.sqrt(np.sum(vectors[:, i] ** 2) * dr)
        if norm > 0:
            vectors[:, i] /= norm

    return energies.astype(float), vectors.astype(float), r.astype(float)


def quantum_dot_gap(
    R_m: float,
    material: dict[str, Any],
    V0_eV: float = 2.0,
    n_points: int = 400,
    l: int = 0,
) -> dict[str, Any]:
    """
    Optical gap from a finite-well electron + hole calculation (eV).

    E_gap = Eg + E_e + E_h − Coulomb(R)

    Electron and hole ground states (given l) are solved separately with the
    same band-offset depth V0.
    """
    V0_J = float(eV_to_J(V0_eV))
    me = effective_mass_kg(material["m_e"])
    mh = effective_mass_kg(material["m_h"])

    e_energies, e_wf, r_grid = finite_spherical_well_energies(
        R_m, me, V0_J, n_points=n_points, l=l, n_states=3
    )
    h_energies, h_wf, _ = finite_spherical_well_energies(
        R_m, mh, V0_J, n_points=n_points, l=l, n_states=3
    )

    E_e_eV = float(J_to_eV(e_energies[0]))
    E_h_eV = float(J_to_eV(h_energies[0]))
    coulomb_eV = coulomb_term(R_m, material["epsilon"])
    gap_eV = float(material["Eg"] + E_e_eV + E_h_eV - coulomb_eV)

    return {
        "gap_eV": gap_eV,
        "E_e_eV": E_e_eV,
        "E_h_eV": E_h_eV,
        "coulomb_eV": coulomb_eV,
        "Eg": float(material["Eg"]),
        "e_energies_eV": J_to_eV(e_energies),
        "h_energies_eV": J_to_eV(h_energies),
        "e_wavefunctions": e_wf,
        "h_wavefunctions": h_wf,
        "r_m": r_grid,
        "V0_eV": V0_eV,
    }


def relative_error_vs_infinite(
    R_m: float,
    m_eff_kg: float,
    V0_eV: float = 1.0,
    n_points: int = 500,
    l: int = 0,
    hard_wall: bool = True,
) -> float:
    """
    Relative error of the numerical ground state vs the infinite-well analytic
    result for l = 0.

    Parameters
    ----------
    hard_wall :
        If True (default), the domain ends at r = R with u(R) = 0, which is the
        discrete infinite spherical well (V0 is unused physically).
        If False, use a deep finite well (large V0_eV) with r_max = 3R and
        measure how closely it approaches the infinite limit.
    """
    if l != 0:
        raise ValueError("Analytic infinite-well reference implemented for l = 0 only")
    V0_J = float(eV_to_J(max(V0_eV, 1.0)))
    if hard_wall:
        energies, _, _ = finite_spherical_well_energies(
            R_m,
            m_eff_kg,
            V0_J,
            n_points=n_points,
            l=0,
            r_max_factor=1.0,
            n_states=1,
        )
    else:
        energies, _, _ = finite_spherical_well_energies(
            R_m,
            m_eff_kg,
            V0_J,
            n_points=n_points,
            l=0,
            r_max_factor=3.0,
            n_states=1,
        )
    E_num = float(energies[0])
    E_ana = infinite_well_energy(R_m, m_eff_kg, n=1)
    return abs(E_num - E_ana) / abs(E_ana)


# ---------------------------------------------------------------------------
# Emission wavelength and colour
# ---------------------------------------------------------------------------

def emission_wavelength_nm(E_eV: float) -> float:
    """Convert photon energy (eV) to vacuum wavelength (nm): λ = hc / E."""
    if E_eV <= 0:
        return float("nan")
    wavelength_m = H_PLANCK * C_LIGHT / float(eV_to_J(E_eV))
    return float(m_to_nm(wavelength_m))


def wavelength_to_rgb(wavelength_nm: float) -> tuple[float, float, float]:
    """
    Map a visible wavelength (nm) to an approximate sRGB triple in [0, 1].

    Based on a common piecewise approximation of the CIE colour-matching
    response (Dan Bruton / similar educational implementations). Outside the
    visible window the result fades to near-black.
    """
    wl = float(wavelength_nm)
    if np.isnan(wl):
        return (0.0, 0.0, 0.0)

    if wl < 380.0 or wl > 780.0:
        return (0.05, 0.05, 0.05)

    if wl < 440.0:
        r, g, b = -(wl - 440.0) / (440.0 - 380.0), 0.0, 1.0
    elif wl < 490.0:
        r, g, b = 0.0, (wl - 440.0) / (490.0 - 440.0), 1.0
    elif wl < 510.0:
        r, g, b = 0.0, 1.0, -(wl - 510.0) / (510.0 - 490.0)
    elif wl < 580.0:
        r, g, b = (wl - 510.0) / (580.0 - 510.0), 1.0, 0.0
    elif wl < 645.0:
        r, g, b = 1.0, -(wl - 645.0) / (645.0 - 580.0), 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0

    # Intensity fall-off near the edges of vision.
    if wl < 420.0:
        factor = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0)
    elif wl > 700.0:
        factor = 0.3 + 0.7 * (780.0 - wl) / (780.0 - 700.0)
    else:
        factor = 1.0

    # Gamma correction for display.
    gamma = 0.8
    r = (r * factor) ** gamma
    g = (g * factor) ** gamma
    b = (b * factor) ** gamma
    return (float(np.clip(r, 0, 1)), float(np.clip(g, 0, 1)), float(np.clip(b, 0, 1)))


def confinement_regime(R_nm: float, a_B_nm: float) -> str:
    """
    Classify confinement strength from R / a_B.

    Strong: R << a_B, intermediate: R ~ a_B, weak: R >> a_B.
    Thresholds: strong if R/a_B < 0.5, weak if R/a_B > 2, else intermediate.
    """
    ratio = float(R_nm) / float(a_B_nm)
    if ratio < 0.5:
        return "fuerte"
    if ratio > 2.0:
        return "débil"
    return "intermedio"


def radius_grid_m(
    r_min_nm: float = 1.0,
    r_max_nm: float = 10.0,
    n: int = 40,
) -> np.ndarray:
    """Uniform radius grid in metres for gap-vs-size sweeps."""
    return nm_to_m(np.linspace(r_min_nm, r_max_nm, n)).astype(float)
