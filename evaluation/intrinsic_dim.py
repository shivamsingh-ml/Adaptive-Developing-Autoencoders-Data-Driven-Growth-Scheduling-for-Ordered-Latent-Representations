import numpy as np
from sklearn.neighbors import NearestNeighbors


def twonn_intrinsic_dim(x, eps=1e-12):
    nn = NearestNeighbors(n_neighbors=3)
    nn.fit(x)
    distances, _ = nn.kneighbors(x)

    r1 = distances[:, 1] + eps
    r2 = distances[:, 2] + eps

    mu = r2 / r1
    mu = np.sort(mu)

    n = len(mu)
    f = np.arange(1, n + 1) / (n + 1)
    x_axis = np.log(mu)
    y_axis = -np.log(1 - f)

    denom = np.sum(x_axis ** 2)
    if denom < eps or np.any(np.isnan(x_axis)):
        return 1.0   # fallback: assume 1D when degenerate
    slope = np.sum(x_axis * y_axis) / denom
    return float(slope)


def twonn_intrinsic_dim_exact(data):
    """Authors' Mottes implementation — matches paper. O(N^2), use for final eval."""
    data = np.array(data)
    N = len(data)
    mu = []
    for i, x in enumerate(data):
        dist = np.sort(np.sqrt(np.sum((x - data)**2, axis=1)))
        r1, r2 = dist[dist > 0][:2]
        mu.append((i+1, r2/r1))
    sigma_i = dict(zip(range(1, len(mu)+1),
                       np.array(sorted(mu, key=lambda x: x[1]))[:, 0].astype(int)))
    mu = dict(mu)
    F_i = {sigma_i[i]: i/N for i in mu}
    x = np.log([mu[i] for i in sorted(mu.keys())])
    y = np.array([1 - F_i[i] for i in sorted(mu.keys())])
    x, y = x[y > 0], y[y > 0]
    y = -np.log(y)
    d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]
    return float(d)