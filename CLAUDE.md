# Visual-Deep-Learning

Publication-quality deep learning visualizations.

## Structure
- `lib/` — shared utilities (theme, smoothing, IO)
- `loss_curve/` — training/loss curve plots
- `tsne/` — t-SNE embedding maps
- `umap_viz/` — UMAP projections
- `embedding_drift/` — before/after embedding comparison (Procrustes)
- `attention/` — attention heatmaps
- `tests/` — pytest tests

## Conventions
- Every module has `plot.py` as its main entry point
- Examples go in `<module>/examples/`
- All plots use `lib.theme.apply_theme()` for consistent styling
- Prefer returning `(fig, ax)` tuples so callers can customize
- Type hints on all public functions
- Functions accept numpy arrays; torch tensor conversion is caller's job

## Dependencies
- Core: matplotlib, seaborn, numpy, scipy, scikit-learn, umap-learn
- Optional: torch, transformers (for attention extraction)

## Testing
```bash
pytest tests/
```

## Install
```bash
pip install -e ".[dev]"
```
