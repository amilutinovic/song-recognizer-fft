import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (dft, dft_via_matrix, dft_matrix, fft_recursive, fft,
                  fft_batch, ifft_batch, is_pow2, _bit_reverse_indices)


rng = np.random.default_rng(42)


def random_signal(n, complex_valued=False):
    x = rng.standard_normal(n)
    if complex_valued:
        x = x + 1j * rng.standard_normal(n)
    return x

# Numpy mathches

def test_dft_matches_reference():
    """DFT from the definition, including non-power-of-two lengths."""
    for n in [1, 2, 3, 5, 8, 17, 64]:
        x = random_signal(n, complex_valued=True)
        assert np.allclose(dft(x), np.fft.fft(x), atol=1e-9), f"dft N={n}"


def test_dft_via_matrix_matches_reference():
    for n in [1, 2, 3, 5, 8, 17, 64]:
        x = random_signal(n, complex_valued=True)
        assert np.allclose(dft_via_matrix(x), np.fft.fft(x), atol=1e-9), n


def test_fft_recursive_matches_reference():
    for n in [1, 2, 4, 8, 16, 64, 256, 1024]:
        x = random_signal(n, complex_valued=True)
        assert np.allclose(fft_recursive(x), np.fft.fft(x), atol=1e-9), n


def test_fft_matches_reference():
    for n in [1, 2, 4, 8, 16, 64, 256, 1024]:
        x = random_signal(n, complex_valued=True)
        assert np.allclose(fft(x), np.fft.fft(x), atol=1e-9), n


def test_all_implementations_agree():
    """All four implementations produce the same result."""
    for n in [8, 64, 256]:
        x = random_signal(n)
        ref = np.fft.fft(x)
        assert np.allclose(dft(x), ref, atol=1e-9)
        assert np.allclose(dft_via_matrix(x), ref, atol=1e-9)
        assert np.allclose(fft_recursive(x), ref, atol=1e-9)
        assert np.allclose(fft(x), ref, atol=1e-9)


def test_fft_batch_matches_loop():
    """The batch version matches calling fft() row by row."""
    frames = random_signal(17 * 256).reshape(17, 256)
    batch = fft_batch(frames)
    loop = np.stack([fft(row) for row in frames])
    assert np.allclose(batch, loop, atol=1e-9)
    assert np.allclose(batch, np.fft.fft(frames, axis=1), atol=1e-9)


def test_ifft_inverts_fft():
    """ifft_batch undoes fft_batch: ifft(fft(x)) == x."""
    x = random_signal(256)
    X = fft_batch(x.reshape(1, -1))
    recovered = ifft_batch(X)[0]
    assert np.allclose(recovered, x, atol=1e-9)


def test_ifft_matches_numpy():
    X = random_signal(128, complex_valued=True).reshape(1, -1)
    assert np.allclose(ifft_batch(X)[0], np.fft.ifft(X[0]), atol=1e-9)


# Helpers and Error handling

def test_bit_reverse_known_case():
    """N=8: 0,1,2,3,4,5,6,7 -> 0,4,2,6,1,5,3,7"""
    assert list(_bit_reverse_indices(8)) == [0, 4, 2, 6, 1, 5, 3, 7]


def test_is_pow2():
    for n, expected in [(0, False), (1, True), (2, True), (3, False), (1024, True)]:
        assert is_pow2(n) is expected


def test_fft_rejects_non_pow2():
    raised = False
    try:
        fft(random_signal(100))
    except ValueError:
        raised = True
    assert raised, "fft should reject a non-power-of-two length"


def test_dft_matrix_is_unitary_up_to_scale():
    """W @ conj(W).T = N * I"""
    n = 32
    W = dft_matrix(n)
    assert np.allclose(W @ np.conj(W).T, n * np.eye(n), atol=1e-9)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]

    passed = 0
    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except AssertionError as err:
            print(f"  [FAIL] {test.__name__}  ->  {err}")
        except Exception as err:
            print(f"  [ERROR] {test.__name__}  ->  {err}")

    print(f"\n{passed}/{len(tests)} tests passed.")
    if passed == len(tests):
        print("All Fourier core tests passed.")