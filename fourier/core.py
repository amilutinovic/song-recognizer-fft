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


#Iterative FFT
##################################

def _bit_reverse_indices(N: int) -> np.ndarray:
    """
    Return the bit-reversal permutation of indices 0 .. N-1.

    The recursive FFT keeps splitting into even/odd
    indices. If you follow where each sample ends up, the order turns
    out to be exactly the bit-reversed order. The iterative version has
    no recursion to do that splitting, so it reorders the samples up
    front, then builds the result bottom-up.
    """
    bits = N.bit_length() - 1        # e.g. N=8 -> 3 bits
    idx = np.arange(N)
    rev = np.zeros(N, dtype=int)
    for i in range(bits):
        # Take bit i of every index and place it at the mirrored position
        rev = (rev << 1) | ((idx >> i) & 1)
    return rev

def fft(x: np.ndarray) -> np.ndarray:
    """Iterative radix-2 FFT, vectorized over NumPy arrays.

    Instead of recursion, it works bottom-up: first the samples are
    reordered by the bit-reversal permutation, then in log2(N) passes
    the blocks of length 2, 4, 8, ... N are merged.

    In a single pass *all* butterflies at that level
    are computed at once, as an operation over an array of shape
    (num_blocks, block_size). Hence there is no Python loop over blocks,
    only log2(N) passes in total.

    This is the default implementation used by the rest of the project.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("FFT expects a 1D array; use fft_batch for many signals")
    return fft_batch(x.reshape(1, -1))[0]


def fft_batch(frames: np.ndarray) -> np.ndarray:
    """FFT over every row of a 2D array (frames.shape = (num_frames, N)).

    This is the shape the STFT layer needs: all windows of the signal
    are transformed with a single call, without a Python loop over
    frames. As a result, processing a whole song is an order of
    magnitude faster than calling fft() in a loop.
    """
    frames = np.asarray(frames)
    if frames.ndim == 1:
        frames = frames.reshape(1, -1)
    if frames.ndim != 2:
        raise ValueError("fft_batch expects a 1D or 2D array")

    n_frames, N = frames.shape
    if not is_pow2(N):
        raise ValueError(
            f"Frame length must be a power of two, got N={N} "
        )

    # Bit-reversal reordering of all frames at once.
    X = frames[:, _bit_reverse_indices(N)].astype(np.complex128, copy=True)

    # 2) At each pass the block size doubles.
    size = 2
    while size <= N:
        half = size // 2
        # Twiddle factors for the current level: w^j, j = 0 .. half-1
        w = np.exp(-2j * np.pi * np.arange(half) / size)

        # Reshape to (n_frames, num_blocks, size) 
        blocks = X.reshape(n_frames, -1, size)

        # The butterfly is computed in-place: the upper half of the
        # block is multiplied by the twiddle factors, then added to and
        # subtracted from the lower half.
        t = blocks[:, :, half:] * w
        blocks[:, :, half:] = blocks[:, :, :half] - t
        blocks[:, :, :half] += t

        size <<= 1

    return X


def ifft_batch(X: np.ndarray) -> np.ndarray:
    """
    Inverse DFT/IFFT computed directly from its definition.

    X.shape = (num_frames, N)

    For every frame:

        x[n] = (1/N) * sum_{k=0}^{N-1}X[k] * exp(2*pi*i*k*n/N)

    Returns:
        Array of shape (num_frames, N).
    """
    X = np.asarray(X, dtype=complex)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    num_frames, N = X.shape


    # Frequency indices
    k = np.arange(N)

    # Time indices
    n = np.arange(N)

    # IFFT matrix: W[n, k] = exp(2*pi*i*k*n/N)
    W = np.exp(2j * np.pi * np.outer(n, k) / N)

    return (X @ W.T) / N
