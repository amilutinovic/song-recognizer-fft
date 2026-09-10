"""
Experiment: STFT with vs without a window function.

Shows what happens when we skip the analysis window (i.e. use a rectangular
window = all ones).

Run from the project experiment:
    python experiment_window.py
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fourier"))

from core import fft_batch
from window import get_window


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = 1024
    fs = 11025

    # A tone whose frequency does NOT sit exactly on a bin, so leakage shows.
    # Bin spacing is fs/N. We pick 40.5 bins -> deliberately between two bins.
    freq = 40.5 * fs / N
    n = np.arange(N)
    signal = np.sin(2 * np.pi * freq * n / fs)

    # Spectrum WITHOUT a window (rectangular = all ones).
    rect = get_window("rect", N)
    spec_rect = np.abs(fft_batch((signal * rect).reshape(1, -1))[0])[: N // 2]

    # Spectrum WITH a Hann window.
    hann = get_window("hann", N)
    spec_hann = np.abs(fft_batch((signal * hann).reshape(1, -1))[0])[: N // 2]

    # Convert to dB, normalized to 0 dB peak, for a fair comparison.
    def to_db(x):
        x = np.maximum(x, 1e-12)
        return 20 * np.log10(x / x.max())

    freqs = np.arange(N // 2) * fs / N

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(freqs, to_db(spec_rect), label="without window (rectangular)",
            color="crimson")
    ax.plot(freqs, to_db(spec_hann), label="with Hann window", color="teal")
    ax.axvline(freq, color="gray", linestyle=":", alpha=0.6,
               label=f"true tone ({freq:.0f} Hz)")
    ax.set_xlim(freq - 600, freq + 600)
    ax.set_ylim(-100, 5)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("magnitude [dB]")
    ax.set_title("Spectral leakage: with vs without a window function")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ROOT, "window_experiment.png")
    fig.savefig(out, dpi=150)

    # Numbers for the report: how wide is the peak / how much energy leaks?
    def leakage_floor(db):
        # Average level of everything more than 5 bins away from the peak
        peak = np.argmax(db)
        mask = np.abs(np.arange(len(db)) - peak) > 5
        return np.mean(db[mask])

    print(f"true tone frequency: {freq:.1f} Hz (between bins 40 and 41)")
    print(f"leakage floor without window: {leakage_floor(to_db(spec_rect)):6.1f} dB")
    print(f"leakage floor with Hann:      {leakage_floor(to_db(spec_hann)):6.1f} dB")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()