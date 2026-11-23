# Efecto del tamaño en puntos cuánticos semiconductores

Aplicación para explorar el confinamiento cuántico en nanocrystales: modelo de Brus, pozo esférico finito (Schrödinger radial con diferencias finitas) y color de emisión estimado.

## Requisitos

- Python **3.11**
- Dependencias en `requirements.txt`

## Instalación

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecutar la app

Desde la raíz del proyecto:

```bash
streamlit run app.py
```

Controles en la barra lateral: material (CdSe, CdS, InP, PbS), radio R, profundidad del pozo V₀ y resolución de la malla.

### Pestañas

1. **Confinamiento** — pozo de potencial, niveles y |u(r)|²; gap, λ y régimen R/a_B.
2. **Gap contra tamaño** — Brus vs solver numérico vs Eg bulk (+ puntos experimentales de CdSe).
3. **Color del punto cuántico** — viales coloreados por λ de emisión.
4. **Validación** — error frente al pozo infinito y convergencia con `n_points`.

## Tests

```bash
pytest -q
```

## Estructura

| Archivo | Rol |
|---------|-----|
| `app.py` | Entrada Streamlit |
| `physics.py` | Brus, solver tridiagonal, λ ↔ RGB |
| `materials.py` | Parámetros de materiales y datos experimentales |
| `plots.py` | Figuras Plotly |
| `tests/test_physics.py` | Validación numérica |
