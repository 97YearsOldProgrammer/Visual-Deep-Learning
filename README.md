# Visual-Deep-Learning

Publication-quality visualizations for deep learning research.

## Modules

| Module | Description |
|--------|-------------|
| `loss_curve` | Training/validation loss curves with smoothing |
| `tsne` | t-SNE embedding maps |
| `umap_viz` | UMAP projections |
| `embedding_drift` | Before/after embedding comparison with Procrustes alignment |
| `attention` | Attention heatmaps and rollout |

## Install

```bash
pip install -e ".[dev]"
```

For torch-dependent features (attention extraction):
```bash
pip install -e ".[all]"
```

## Quick Start

```python
from lib.theme import apply_theme
from loss_curve.plot import plot_loss

apply_theme()
fig, ax = plot_loss(train_losses, val_losses, smooth=True)
fig.savefig("loss.pdf")
```
