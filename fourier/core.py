"""
Fourier transform core.
"""
from __future__ import annotations

import numpy as np

def dft(x: np.ndarray) -> np.ndarray:
    """
    Direct DFT from the definition.

        X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n / N)

    Complexity O(N^2). This is the slowest way, but it is the clearest translation
    of the mathematical definition.
    """
    x = np.asarray(x, dtype=complex)
    if x.ndim != 1:
        raise ValueError("DFT expects a 1D array")

    N = x.shape[0]
    X = np.zeros(N, dtype=complex)

    for k in range(N):                       
        total = 0.0
        for n in range(N):                   # sum over all input samples n
            angle = -2j * np.pi * k * n / N
            total += x[n] * np.exp(angle)
        X[k] = total

    return X