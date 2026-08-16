"""
Fourier transform core.
"""
from __future__ import annotations

import numpy as np


# Direct DFT
#################################

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

# Matrix form of DFT
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

# Recursive Cooley-Tukey FFT
############################################

def is_pow2(n: int) -> bool:
    """
    True if n is a power of two and n > 0.
    """
    return n > 0 and (n & (n - 1)) == 0


def fft_recursive(x: np.ndarray) -> np.ndarray:
    """
    Recursive radix-2 Cooley-Tukey FFT.  Complexity O(N log N).

    DFT of length N can be split into two DFTs of length N/2 -
    one over the even-indexed samples, one over the odd-indexed ones. 
    Their results combine like this:

        X[k]        = E[k] + w^k * O[k]
        X[k + N/2]  = E[k] - w^k * O[k]      for k = 0 .. N/2 - 1

    where E = FFT(even samples), O = FFT(odd samples), and
    w = exp(-2*pi*i / N) is the "twiddle factor".
    """
    x = np.asarray(x, dtype=complex)
    if x.ndim != 1:
        raise ValueError("fft_recursive expects a 1D array")

    N = x.shape[0]
    if not is_pow2(N):
        raise ValueError(f"Length must be a power of two, got N={N}")

    # Base case: the DFT of a single sample is the sample itself.
    if N == 1:
        return x

    # Split into even and odd indexed samples and transform each half.
    even = fft_recursive(x[0::2])    # x[0], x[2], x[4], ...
    odd = fft_recursive(x[1::2])     # x[1], x[3], x[5], ...

    # Twiddle factors w^k for k = 0 .. N/2 - 1
    k = np.arange(N // 2)
    w = np.exp(-2j * np.pi * k / N)

    # Combine the two halves (the "butterfly").
    t = w * odd
    return np.concatenate([even + t, even - t])

