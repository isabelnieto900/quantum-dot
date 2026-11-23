"""
Material parameters and experimental data for semiconductor quantum dots.

Energies in the dictionaries are stored in eV; masses as fractions of the
free-electron mass m_e; lengths in nm. Conversion to SI happens in physics.py.

Sources are cited in comments next to each field. Missing reliable values are
left as None with a TODO marker — do not invent numbers.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Bulk material parameters
# ---------------------------------------------------------------------------

MATERIALS: dict[str, dict[str, Any]] = {
    "CdSe": {
        # Bulk band gap (wurtzite CdSe at ~300 K).
        # Ninomiya & Adachi, J. Appl. Phys. 78, 1183 (1995); commonly ~1.74 eV.
        "Eg": 1.74,
        # Effective masses (in units of m_e).
        # Norris & Bawendi, Phys. Rev. B 53, 16338 (1996); me* ≈ 0.13, mh* ≈ 0.45.
        "m_e": 0.13,
        "m_h": 0.45,
        # Static dielectric constant.
        # Adachi, "Properties of Group-IV, III-V and II-VI Semiconductors" (~9.5–10.2).
        "epsilon": 9.7,
        # Exciton Bohr radius a_B.
        # Klimov, "Nanocrystal Quantum Dots" / Norris & Bawendi (~5.6 nm).
        "a_B": 5.6,
    },
    "CdS": {
        # Bulk band gap (wurtzite CdS at room temperature) ≈ 2.42 eV.
        # Madelung, "Semiconductors: Data Handbook".
        "Eg": 2.42,
        # Effective masses.
        # Lippens & Lannoo, Phys. Rev. B 39, 10935 (1989): me* ≈ 0.19, mh* ≈ 0.80.
        "m_e": 0.19,
        "m_h": 0.80,
        # Static dielectric constant ≈ 8.9.
        # Madelung / Adachi handbooks.
        "epsilon": 8.9,
        # Exciton Bohr radius ≈ 3.0 nm.
        # Yoffe, Adv. Phys. 42, 173 (1993) and related QD reviews.
        "a_B": 3.0,
    },
    "InP": {
        # Bulk band gap (zincblende InP at 300 K) ≈ 1.34 eV.
        # Vurgaftman et al., J. Appl. Phys. 89, 5815 (2001).
        "Eg": 1.34,
        # Effective masses.
        # Vurgaftman et al.: me* ≈ 0.0795; heavy-hole average often ~0.6 in QD models.
        "m_e": 0.08,
        "m_h": 0.60,
        # Static dielectric constant ≈ 12.5.
        # Vurgaftman et al. / Adachi.
        "epsilon": 12.5,
        # Exciton Bohr radius ≈ 10 nm (weak confinement typical for larger InP QDs).
        # Fu & Zunger, Phys. Rev. B 55, 1642 (1997) and QD reviews (~9–15 nm).
        "a_B": 10.0,
    },
    "PbS": {
        # Bulk band gap (rock-salt PbS at 300 K) ≈ 0.41 eV.
        # Madelung handbook / Kang & Wise, J. Opt. Soc. Am. B 14, 1632 (1997).
        "Eg": 0.41,
        # Effective masses (near-isotropic; often taken equal for e and h).
        # Kang & Wise: me* ≈ mh* ≈ 0.085.
        "m_e": 0.085,
        "m_h": 0.085,
        # Optical/static dielectric constant used in Brus-type models ≈ 17.2.
        # Kang & Wise (1997).
        "epsilon": 17.2,
        # Exciton Bohr radius ≈ 18–20 nm (strong confinement even at large R).
        # Kang & Wise (1997) report a_B ≈ 18 nm.
        "a_B": 18.0,
    },
}


# ---------------------------------------------------------------------------
# Experimental gap vs size
# Diameter (nm) → optical gap (eV) from the first excitonic absorption peak
# via E(eV) = hc/λ ≈ 1239.8 / λ(nm).
#
# CdSe points are taken from the sizing / absorption data associated with
# Yu et al., Chem. Mater. 2003, 15, 2854–2860 (Table 1 / sizing curve context).
# Wavelengths below are the reported first-exciton peaks; gaps are derived.
# ---------------------------------------------------------------------------

EXPERIMENTAL_GAP: dict[str, list[tuple[float, float]]] = {
    # (diameter_nm, gap_eV)
    "CdSe": [
        # λ ≈ 474 nm → 2.615 eV; D ≈ 2.2 nm (Yu et al. 2003 sizing context)
        (2.2, 2.615),
        # λ ≈ 500 nm → 2.480 eV; D ≈ 2.5 nm
        (2.5, 2.480),
        # λ ≈ 520 nm → 2.384 eV; D ≈ 2.8 nm
        (2.8, 2.384),
        # λ ≈ 540 nm → 2.296 eV; D ≈ 3.1 nm
        (3.1, 2.296),
        # λ ≈ 560 nm → 2.214 eV; D ≈ 3.5 nm
        (3.5, 2.214),
        # λ ≈ 580 nm → 2.138 eV; D ≈ 4.0 nm
        (4.0, 2.138),
        # λ ≈ 600 nm → 2.066 eV; D ≈ 4.6 nm
        (4.6, 2.066),
        # λ ≈ 620 nm → 2.000 eV; D ≈ 5.2 nm
        (5.2, 2.000),
    ],
    # TODO: add published gap-vs-diameter points for CdS (e.g. Yu et al. 2003 CdS set).
    "CdS": [],
    # TODO: add published InP QD gap-vs-size points from a primary source.
    "InP": [],
    # TODO: add published PbS QD gap-vs-size points (e.g. Moreels / Kang & Wise data).
    "PbS": [],
}


def get_material(name: str) -> dict[str, Any]:
    """Return a copy of the material parameter dict for *name*."""
    if name not in MATERIALS:
        known = ", ".join(sorted(MATERIALS))
        raise KeyError(f"Unknown material '{name}'. Known: {known}")
    return dict(MATERIALS[name])


def list_materials() -> list[str]:
    """Return sorted material names."""
    return sorted(MATERIALS.keys())
