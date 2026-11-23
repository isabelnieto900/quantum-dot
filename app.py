"""
Streamlit app: size effect in semiconductor quantum dots.

Run with:  streamlit run app.py

UI language: Spanish. Code and physics comments: English.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from materials import EXPERIMENTAL_GAP, get_material, list_materials
from physics import (
    J_to_eV,
    brus_gap,
    confinement_regime,
    effective_mass_kg,
    emission_wavelength_nm,
    eV_to_J,
    finite_spherical_well_energies,
    infinite_well_energy,
    nm_to_m,
    quantum_dot_gap,
    relative_error_vs_infinite,
    wavelength_to_rgb,
)
from plots import (
    plot_gap_vs_size,
    plot_mesh_convergence,
    plot_potential_and_states,
    plot_validation_error,
    plot_vial_row,
)

st.set_page_config(
    page_title="Puntos cuánticos — efecto del tamaño",
    page_icon="🔬",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached physics wrappers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def cached_qd_gap(
    material_name: str,
    R_nm: float,
    V0_eV: float,
    n_points: int,
    l: int,
) -> dict:
    """Cached finite-well quantum-dot gap for a single radius."""
    material = get_material(material_name)
    return quantum_dot_gap(
        float(nm_to_m(R_nm)),
        material,
        V0_eV=V0_eV,
        n_points=n_points,
        l=l,
    )


@st.cache_data(show_spinner=False)
def cached_gap_curves(
    material_name: str,
    V0_eV: float,
    n_points: int,
    r_min: float,
    r_max: float,
    n_radii: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Cached Brus + numerical gap curves vs radius."""
    material = get_material(material_name)
    radii_nm = np.linspace(r_min, r_max, n_radii)
    brus = np.empty_like(radii_nm)
    numeric = np.empty_like(radii_nm)
    for i, R in enumerate(radii_nm):
        R_m = float(nm_to_m(R))
        brus[i] = brus_gap(R_m, material)
        numeric[i] = quantum_dot_gap(R_m, material, V0_eV=V0_eV, n_points=n_points)["gap_eV"]
    return radii_nm, brus, numeric, float(material["Eg"])


@st.cache_data(show_spinner=False)
def cached_validation_curve(
    material_name: str,
    n_points: int,
    r_min: float,
    r_max: float,
    n_radii: int,
    hard_wall: bool,
    V0_eV: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cached relative error vs infinite well for the electron mass."""
    material = get_material(material_name)
    me = effective_mass_kg(material["m_e"])
    radii_nm = np.linspace(r_min, r_max, n_radii)
    errors = np.empty_like(radii_nm)
    for i, R in enumerate(radii_nm):
        errors[i] = relative_error_vs_infinite(
            float(nm_to_m(R)),
            me,
            V0_eV=V0_eV,
            n_points=n_points,
            l=0,
            hard_wall=hard_wall,
        )
    return radii_nm, errors


@st.cache_data(show_spinner=False)
def cached_mesh_convergence(
    material_name: str,
    R_nm: float,
    n_list: tuple[int, ...],
) -> tuple[list[int], list[float], float]:
    """Ground-state energy vs mesh size for the hard-wall (infinite) well."""
    material = get_material(material_name)
    me = effective_mass_kg(material["m_e"])
    R_m = float(nm_to_m(R_nm))
    # V0 is unused when r_max_factor=1 (hard wall); pass a dummy positive value.
    V0_J = float(eV_to_J(1.0))
    energies = []
    for n in n_list:
        E_J, _, _ = finite_spherical_well_energies(
            R_m, me, V0_J, n_points=n, l=0, r_max_factor=1.0, n_states=1
        )
        energies.append(float(J_to_eV(E_J[0])))
    E_inf = float(J_to_eV(infinite_well_energy(R_m, me, n=1)))
    return list(n_list), energies, E_inf


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.title("Efecto del tamaño en puntos cuánticos semiconductores")
st.caption(
    "Modelo de Brus y resolución numérica de la ecuación de Schrödinger "
    "en un pozo esférico finito (masas efectivas)."
)

with st.sidebar:
    st.header("Parámetros")
    material_name = st.selectbox("Material", list_materials(), index=0)
    R_nm = st.slider("Radio R (nm)", min_value=1.0, max_value=10.0, value=2.5, step=0.1)
    V0_eV = st.slider(
        "Profundidad del pozo V₀ (eV)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Offset de bandas (misma profundidad para electrón y hueco).",
    )
    n_points = st.select_slider(
        "Puntos de malla",
        options=[100, 150, 200, 300, 400, 500, 600, 800],
        value=400,
    )
    l_quantum = st.radio(
        "Momento angular l",
        options=[0, 1],
        index=0,
        horizontal=True,
        help="Se resuelve por separado para electrón y hueco.",
    )

    material = get_material(material_name)
    st.markdown("---")
    st.markdown(
        f"**{material_name}**  \n"
        f"Eg = {material['Eg']:.2f} eV  \n"
        f"mₑ* = {material['m_e']:.3f} mₑ  \n"
        f"mₕ* = {material['m_h']:.3f} mₑ  \n"
        f"ε = {material['epsilon']:.1f}  \n"
        f"a_B = {material['a_B']:.1f} nm"
    )


# ---------------------------------------------------------------------------
# Live QD calculation for current radius
# ---------------------------------------------------------------------------

with st.spinner("Calculando estados del pozo finito…"):
    result = cached_qd_gap(material_name, R_nm, V0_eV, n_points, l_quantum)

gap_eV = result["gap_eV"]
lambda_nm = emission_wavelength_nm(gap_eV)
ratio = R_nm / material["a_B"]
regime = confinement_regime(R_nm, material["a_B"])
rgb = wavelength_to_rgb(lambda_nm)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_conf, tab_gap, tab_color, tab_val = st.tabs(
    [
        "1. Confinamiento",
        "2. Gap contra tamaño",
        "3. Color del punto cuántico",
        "4. Validación",
    ]
)

with tab_conf:
    st.subheader("Pozo de potencial y funciones de onda")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gap óptico", f"{gap_eV:.3f} eV")
    m2.metric("λ de emisión", f"{lambda_nm:.1f} nm")
    m3.metric("R / a_B", f"{ratio:.2f}")
    m4.metric("Régimen", regime)

    fig_well = plot_potential_and_states(
        result["r_m"],
        result["V0_eV"],
        float(nm_to_m(R_nm)),
        result["e_energies_eV"],
        result["h_energies_eV"],
        result["e_wavefunctions"],
        result["h_wavefunctions"],
        n_show=2,
    )
    st.plotly_chart(fig_well, use_container_width=True)

    st.markdown(
        f"Eₑ = **{result['E_e_eV']:.4f} eV**, "
        f"Eₕ = **{result['E_h_eV']:.4f} eV**, "
        f"término de Coulomb = **{result['coulomb_eV']:.4f} eV**.  \n"
        f"E_gap = Eg + Eₑ + Eₕ − Coulomb."
    )

with tab_gap:
    st.subheader("Curvas de gap frente al radio")
    with st.spinner("Calculando curvas Brus / numéricas…"):
        radii_nm, brus_vals, num_vals, Eg = cached_gap_curves(
            material_name, V0_eV, n_points, 1.0, 10.0, 36
        )

    experimental = EXPERIMENTAL_GAP.get(material_name, [])
    fig_gap = plot_gap_vs_size(
        radii_nm,
        brus_vals,
        num_vals,
        Eg,
        R_nm,
        gap_eV,
        experimental=experimental,
    )
    st.plotly_chart(fig_gap, use_container_width=True)

    if experimental:
        st.caption(
            "Puntos experimentales de CdSe: gap óptico vs diámetro publicado "
            "(convertido a radio = D/2) según Yu et al., Chem. Mater. 2003, 15, 2854–2860."
        )
    else:
        st.info(
            f"No hay puntos experimentales cargados para {material_name} "
            "(ver TODO en materials.py)."
        )

with tab_color:
    st.subheader("Color de emisión estimado")
    st.write(
        "Cada vial se colorea con el RGB correspondiente a λ = hc/E_gap "
        "del solver numérico en ese radio."
    )

    animate = st.checkbox("Animar barrido de radio", value=False)
    vial_radii = np.linspace(1.5, 6.0, 8)

    if animate:
        anim_R = st.slider(
            "Radio animado (nm)",
            min_value=float(vial_radii[0]),
            max_value=float(vial_radii[-1]),
            value=float(R_nm) if vial_radii[0] <= R_nm <= vial_radii[-1] else float(vial_radii[0]),
            step=0.1,
            key="anim_radius",
        )
        highlight = int(np.argmin(np.abs(vial_radii - anim_R)))
        # Recompute gap at animated radius for the metric row.
        anim_result = cached_qd_gap(material_name, anim_R, V0_eV, n_points, l_quantum)
        anim_lam = emission_wavelength_nm(anim_result["gap_eV"])
        st.metric("λ (radio animado)", f"{anim_lam:.1f} nm")
    else:
        highlight = int(np.argmin(np.abs(vial_radii - R_nm)))

    colors = []
    wavelengths = []
    for Rv in vial_radii:
        res_v = cached_qd_gap(material_name, float(Rv), V0_eV, n_points, l_quantum)
        lam = emission_wavelength_nm(res_v["gap_eV"])
        wavelengths.append(lam)
        colors.append(wavelength_to_rgb(lam))

    fig_vials = plot_vial_row(vial_radii, colors, wavelengths, highlight_idx=highlight)
    st.plotly_chart(fig_vials, use_container_width=True)

    swatch = (
        f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
    )
    st.markdown(
        f"Color del radio actual (R = {R_nm:.2f} nm, λ ≈ {lambda_nm:.1f} nm): "
        f"<span style='display:inline-block;width:1.4em;height:1.4em;"
        f"border-radius:50%;background:{swatch};border:1px solid #333;"
        f"vertical-align:middle;'></span>",
        unsafe_allow_html=True,
    )

with tab_val:
    st.subheader("Solver frente al pozo infinito")
    st.write(
        "Referencia analítica (l = 0): E = ħ²π²/(2 m R²), masa efectiva del electrón. "
        "Por defecto se usa pared dura en r = R (discretización del pozo infinito)."
    )
    mode = st.radio(
        "Modo de comparación",
        options=["Pared dura (r_max = R)", "Pozo finito profundo"],
        index=0,
        horizontal=True,
    )
    hard_wall = mode.startswith("Pared")
    V0_val = 1.0
    if not hard_wall:
        V0_val = st.slider(
            "V₀ para validación (eV)",
            min_value=5.0,
            max_value=80.0,
            value=40.0,
            step=1.0,
            key="V0_validation",
        )

    with st.spinner("Calculando error relativo…"):
        r_val, err_val = cached_validation_curve(
            material_name, n_points, 1.0, 10.0, 20, hard_wall, V0_val
        )
    fig_err = plot_validation_error(r_val, err_val, V0_val if not hard_wall else float("inf"))
    st.plotly_chart(fig_err, use_container_width=True)

    me = effective_mass_kg(material["m_e"])
    err_now = relative_error_vs_infinite(
        float(nm_to_m(R_nm)),
        me,
        V0_eV=V0_val,
        n_points=n_points,
        l=0,
        hard_wall=hard_wall,
    )
    st.metric(
        f"Error relativo en R = {R_nm:.2f} nm",
        f"{100.0 * err_now:.3f} %",
    )

    st.subheader("Convergencia con la malla")
    st.caption("Discretización de pared dura: el error debe bajar al aumentar n_points.")
    n_list = (100, 150, 200, 300, 400, 500, 600, 800)
    with st.spinner("Barrido de n_points…"):
        n_used, E_mesh, E_ref = cached_mesh_convergence(material_name, R_nm, n_list)
    fig_mesh = plot_mesh_convergence(n_used, E_mesh, E_ref)
    st.plotly_chart(fig_mesh, use_container_width=True)
