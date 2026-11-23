"""
Plotly figures for the quantum-dot Streamlit app.

All user-facing labels are in Spanish; function names and comments in English.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from physics import m_to_nm


def plot_potential_and_states(
    r_m: np.ndarray,
    V0_eV: float,
    R_m: float,
    e_energies_eV: np.ndarray,
    h_energies_eV: np.ndarray,
    e_wavefunctions: np.ndarray,
    h_wavefunctions: np.ndarray,
    n_show: int = 2,
) -> go.Figure:
    """
    Potential well with energy levels and |u(r)|² overlays for electron and hole.
    """
    r_nm = m_to_nm(r_m)
    R_nm = float(m_to_nm(R_m))
    r_max_nm = float(r_nm[-1])

    # Piecewise potential for display.
    V_e = np.where(r_nm < R_nm, 0.0, V0_eV)
    V_h = V_e.copy()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Electrón", "Hueco"),
        shared_yaxes=False,
    )

    scale_e = 0.35 * V0_eV
    scale_h = 0.35 * V0_eV

    # Electron panel
    fig.add_trace(
        go.Scatter(
            x=r_nm,
            y=V_e,
            mode="lines",
            name="V(r)",
            line=dict(color="#444444", width=2),
            legendgroup="V",
        ),
        row=1,
        col=1,
    )
    for i in range(min(n_show, len(e_energies_eV))):
        E = float(e_energies_eV[i])
        density = e_wavefunctions[:, i] ** 2
        density = density / (density.max() + 1e-30) * scale_e
        fig.add_trace(
            go.Scatter(
                x=r_nm,
                y=E + density,
                mode="lines",
                name=f"|u_e|² n={i + 1}",
                line=dict(width=2),
                legendgroup=f"e{i}",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=E,
            line_dash="dot",
            line_color="rgba(30,100,200,0.7)",
            row=1,
            col=1,
            annotation_text=f"E_e={E:.3f} eV",
            annotation_position="right",
        )

    # Hole panel
    fig.add_trace(
        go.Scatter(
            x=r_nm,
            y=V_h,
            mode="lines",
            name="V(r)",
            line=dict(color="#444444", width=2),
            showlegend=False,
            legendgroup="V",
        ),
        row=1,
        col=2,
    )
    for i in range(min(n_show, len(h_energies_eV))):
        E = float(h_energies_eV[i])
        density = h_wavefunctions[:, i] ** 2
        density = density / (density.max() + 1e-30) * scale_h
        fig.add_trace(
            go.Scatter(
                x=r_nm,
                y=E + density,
                mode="lines",
                name=f"|u_h|² n={i + 1}",
                line=dict(width=2),
                legendgroup=f"h{i}",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(
            y=E,
            line_dash="dot",
            line_color="rgba(180,60,40,0.7)",
            row=1,
            col=2,
            annotation_text=f"E_h={E:.3f} eV",
            annotation_position="right",
        )

    for col in (1, 2):
        fig.add_vline(
            x=R_nm,
            line_dash="dash",
            line_color="gray",
            row=1,
            col=col,
        )

    fig.update_xaxes(title_text="r (nm)", range=[0, r_max_nm], row=1, col=1)
    fig.update_xaxes(title_text="r (nm)", range=[0, r_max_nm], row=1, col=2)
    fig.update_yaxes(title_text="Energía (eV)", row=1, col=1)
    fig.update_yaxes(title_text="Energía (eV)", row=1, col=2)
    fig.update_layout(
        title=f"Pozo esférico finito (R = {R_nm:.2f} nm, V₀ = {V0_eV:.2f} eV)",
        height=480,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.12),
    )
    return fig


def plot_gap_vs_size(
    radii_nm: np.ndarray,
    brus_eV: np.ndarray,
    numeric_eV: np.ndarray,
    Eg: float,
    current_R_nm: float,
    current_gap_eV: float,
    experimental: Sequence[tuple[float, float]] | None = None,
) -> go.Figure:
    """
    Gap vs radius: Brus, numerical solver, bulk gap, and experimental points.

    Experimental tuples are (diameter_nm, gap_eV) as published; they are
    converted to radius = D/2 for plotting against the radius axis.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=radii_nm,
            y=brus_eV,
            mode="lines",
            name="Brus",
            line=dict(width=2, color="#1f77b4"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=radii_nm,
            y=numeric_eV,
            mode="lines",
            name="Solver numérico",
            line=dict(width=2, color="#ff7f0e"),
        )
    )
    fig.add_hline(
        y=Eg,
        line_dash="dash",
        line_color="#2ca02c",
        annotation_text=f"Eg bulk = {Eg:.2f} eV",
        annotation_position="top left",
    )

    if experimental:
        r_exp = [d / 2.0 for d, _ in experimental]
        e_exp = [g for _, g in experimental]
        fig.add_trace(
            go.Scatter(
                x=r_exp,
                y=e_exp,
                mode="markers",
                name="Experimental (Yu et al. 2003)",
                marker=dict(size=10, symbol="diamond", color="#d62728"),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[current_R_nm],
            y=[current_gap_eV],
            mode="markers",
            name="Radio actual",
            marker=dict(size=14, symbol="x", color="black", line=dict(width=2)),
        )
    )

    fig.update_layout(
        title="Gap óptico frente al radio",
        xaxis_title="Radio R (nm)",
        yaxis_title="Gap (eV)",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_vial_row(
    radii_nm: Sequence[float],
    colors: Sequence[tuple[float, float, float]],
    wavelengths_nm: Sequence[float],
    highlight_idx: int | None = None,
) -> go.Figure:
    """
    Row of coloured circles ("vials") representing emission colour vs radius.
    """
    fig = go.Figure()
    n = len(radii_nm)
    for i, (R, rgb, wl) in enumerate(zip(radii_nm, colors, wavelengths_nm)):
        css = f"rgb({int(rgb[0]*255)},{int(rgb[1]*255)},{int(rgb[2]*255)})"
        size = 48 if highlight_idx is not None and i == highlight_idx else 36
        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[0],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=css,
                    line=dict(
                        width=3 if highlight_idx == i else 1,
                        color="black" if highlight_idx == i else "#888888",
                    ),
                ),
                text=[f"R={R:.1f}<br>{wl:.0f} nm"],
                textposition="bottom center",
                name=f"R={R:.1f} nm",
                showlegend=False,
                hovertemplate=f"R={R:.2f} nm<br>λ≈{wl:.1f} nm<extra></extra>",
            )
        )

    fig.update_layout(
        title="Color de emisión estimado (viales)",
        xaxis=dict(
            visible=False,
            range=[-0.8, n - 0.2],
        ),
        yaxis=dict(visible=False, range=[-1.2, 1.0]),
        height=280,
        template="plotly_white",
        margin=dict(t=50, b=40),
    )
    return fig


def plot_validation_error(
    radii_nm: np.ndarray,
    rel_errors: np.ndarray,
    V0_eV: float,
) -> go.Figure:
    """Relative error of the numerical ground state vs the infinite well."""
    if np.isfinite(V0_eV):
        title = f"Validación: solver vs pozo infinito (V₀ = {V0_eV:.1f} eV)"
    else:
        title = "Validación: pared dura (r_max = R) vs pozo infinito analítico"
    fig = go.Figure(
        data=[
            go.Scatter(
                x=radii_nm,
                y=100.0 * rel_errors,
                mode="lines+markers",
                name="Error relativo",
                line=dict(width=2),
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Radio R (nm)",
        yaxis_title="Error relativo (%)",
        template="plotly_white",
        height=420,
    )
    return fig


def plot_mesh_convergence(
    n_points_list: Sequence[int],
    energies_eV: Sequence[float],
    reference_eV: float,
) -> go.Figure:
    """Convergence of the ground-state energy with mesh density."""
    errors = [abs(e - reference_eV) / abs(reference_eV) * 100.0 for e in energies_eV]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(n_points_list),
            y=list(energies_eV),
            mode="lines+markers",
            name="E₀ numérico",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(n_points_list),
            y=errors,
            mode="lines+markers",
            name="Error vs referencia (%)",
            yaxis="y2",
            line=dict(dash="dot"),
        )
    )
    fig.add_hline(
        y=reference_eV,
        line_dash="dash",
        line_color="green",
        annotation_text=f"E∞ = {reference_eV:.4f} eV",
    )
    fig.update_layout(
        title="Convergencia con el número de puntos de malla",
        xaxis_title="Número de puntos (n_points)",
        yaxis=dict(title="Energía E₀ (eV)"),
        yaxis2=dict(title="Error relativo (%)", overlaying="y", side="right"),
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
