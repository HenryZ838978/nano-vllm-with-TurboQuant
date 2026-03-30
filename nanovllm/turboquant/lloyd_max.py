"""
Lloyd-Max optimal scalar quantizer for the Beta distribution arising from
random rotation of unit-norm vectors. After rotation, each coordinate follows
N(0, 1/d) for practical dimensions (d >= 64).

Reference: TurboQuant (ICLR 2026)
"""

import torch
import math
from scipy import integrate


def solve_lloyd_max(d: int, bits: int, max_iter: int = 200, tol: float = 1e-10) -> torch.Tensor:
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)

    def pdf(x):
        return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(-x * x / (2 * sigma ** 2))

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(max_iter):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < tol:
            break
        centroids = new_centroids

    return torch.tensor(centroids, dtype=torch.float32)
