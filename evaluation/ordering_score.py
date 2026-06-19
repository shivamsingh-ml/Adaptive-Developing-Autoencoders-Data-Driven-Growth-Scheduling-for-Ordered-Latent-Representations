"""
ordering_score.py
Quantifies whether bottleneck units are ordered like PCA components:
early units capture high-variance / low-PC-index structure, later units
capture finer / higher-PC-index structure.

Two complementary metrics (both: higher rho = more ordered):
  - pc_alignment_rho:  Spearman(unit_index, aligned_PC_index)
                       directly measures the paper's Fig 5A claim
  - variance_decay_rho: Spearman(unit_index, -activation_variance)
                       cheap proxy; ordered models have early units
                       with higher activation variance

Both consume the SAME bottleneck activations, so compute once and
derive both. Cost ~1s for N=2000, latent=128. Negligible per run.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def _safe_spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan"), float("nan")
    rho, p = spearmanr(a, b)
    return float(rho), float(p)


def pc_alignment_ordering(encodings, inputs_flat, n_pcs=None):
    """
    encodings   : [N, latent_dim] bottleneck activations
    inputs_flat : [N, D] flattened input images
    Returns Spearman rho between unit index and the input-PC index each
    unit aligns most strongly with. Higher = more ordered.
    """
    encodings = np.asarray(encodings)
    N, latent_dim = encodings.shape
    if n_pcs is None:
        n_pcs = min(latent_dim, inputs_flat.shape[1], 50)

    pca = PCA(n_components=n_pcs)
    pc_proj = pca.fit_transform(inputs_flat)   # [N, n_pcs]

    aligned_pc = np.zeros(latent_dim, dtype=int)
    for u in range(latent_dim):
        unit = encodings[:, u]
        if unit.std() < 1e-10:
            aligned_pc[u] = n_pcs            # dead unit -> last
            continue
        corrs = [abs(np.corrcoef(unit, pc_proj[:, p])[0, 1]) for p in range(n_pcs)]
        aligned_pc[u] = int(np.nanargmax(corrs))

    rho, p = _safe_spearman(np.arange(latent_dim), aligned_pc)
    return {"pc_alignment_rho": rho, "pc_alignment_p": p}


def variance_decay_ordering(encodings):
    """
    Higher rho = more ordered (early units carry more variance).
    Returns Spearman(unit_index, -variance): positive when variance
    decreases with index, matching the PCA-like ordering convention.
    """
    encodings = np.asarray(encodings)
    variances = encodings.var(axis=0)
    rho, p = _safe_spearman(np.arange(len(variances)), -variances)
    return {"variance_decay_rho": rho, "variance_decay_p": p}


def compute_ordering_scores(encodings, inputs_flat, n_pcs=None):
    """Convenience: both metrics from one set of activations."""
    out = {}
    out.update(pc_alignment_ordering(encodings, inputs_flat, n_pcs))
    out.update(variance_decay_ordering(encodings))
    return out