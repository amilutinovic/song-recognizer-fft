import numpy as np
import matplotlib.pyplot as plt

from core import fft
from window import get_window


def generate_sine(frequency, sample_rate, num_samples):
    """Generate a sinusoidal signal."""
    n = np.arange(num_samples)

    return np.sin(
        2.0 * np.pi * frequency * n / sample_rate
    )


def magnitude_spectrum(x, sample_rate):
    """
    Compute the one-sided amplitude spectrum.

    Returns:
        frequencies: frequency values in Hz
        magnitude: one-sided amplitude spectrum
    """
    X = fft(x)
    N = len(x)

    frequencies = np.arange(N // 2) * sample_rate / N

    magnitude = np.abs(X[:N // 2]) / N

    # Convert the two-sided spectrum into a one-sided
    # amplitude spectrum. DC is not doubled.
    magnitude[1:] *= 2.0

    return frequencies, magnitude



def experiment_spectral_leakage():
    """
    Demonstrate spectral leakage.

    The signal frequency is intentionally chosen so that it
    does not fall exactly on an FFT bin.
    """

    sample_rate = 1000
    N = 1024

    # Frequency does not correspond exactly to an FFT bin.
    frequency = 123.4

    signal = generate_sine(
        frequency,
        sample_rate,
        N,
    )

    frequencies, magnitude = magnitude_spectrum(
        signal,
        sample_rate,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        frequencies,
        magnitude,
    )

    plt.title("Spectral Leakage")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")

    # Focus on the area around the signal frequency.
    plt.xlim(80, 170)

    plt.grid(True)

    plt.tight_layout()
    plt.show()


def experiment_window_comparison():
    """
    Compare the spectrum of the same signal using:

        - Rectangular window
        - Hann window
        - Hamming window
        - Blackman window

    The signal frequency is not aligned with an FFT bin,
    so spectral leakage is visible.
    """

    sample_rate = 1000
    N = 1024

    frequency = 123.4

    signal = generate_sine(
        frequency,
        sample_rate,
        N,
    )

    windows = {
        "Rectangular": np.ones(N),
        "Hann": get_window("hann", N),
        "Hamming": get_window("hamming", N),
        "Blackman": get_window("blackman", N),
    }

    plt.figure(figsize=(10, 6))

    for name, window in windows.items():

        # Apply the window.
        windowed_signal = signal * window

        frequencies, magnitude = magnitude_spectrum(
            windowed_signal,
            sample_rate,
        )

        # Normalize the peak to 0 dB.
        magnitude = magnitude / np.max(magnitude)

        magnitude_db = 20.0 * np.log10(
            np.maximum(magnitude, 1e-12)
        )

        plt.plot(
            frequencies,
            magnitude_db,
            label=name,
        )

    plt.title(
        "Influence of Window on Spectral Leakage"
    )

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")

    plt.xlim(80, 170)
    plt.ylim(-100, 5)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":


    experiment_spectral_leakage()

    experiment_window_comparison()