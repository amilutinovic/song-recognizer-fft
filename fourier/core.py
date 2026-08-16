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
        for n in range(N):                   
            angle = -2j * np.pi * k * n / N
            total += x[n] * np.exp(angle)
        X[k] = total

    return X

###############################################

def dft_matrix(N: int) -> np.ndarray:
    """
    Build the DFT kernel matrix W, where

        W[k, n] = exp(-2*pi*i*k*n / N)

    Multiplying X = W @ x then computes the DFT in one matrix-vector
    product. Still O(N^2) in arithmetic.
    """
    n = np.arange(N)                 
    k = n.reshape((N, 1))            
    return np.exp(-2j * np.pi * k * n / N)   # outer product -> N x N matrix


def dft_via_matrix(x: np.ndarray) -> np.ndarray:
    """Same result as dft(), but computed as W @ x instead of two loops."""
    x = np.asarray(x, dtype=complex)
    if x.ndim != 1:
        raise ValueError("dft_via_matrix expects a 1D array")
    N = x.shape[0]
    return dft_matrix(N) @ x

############################################
