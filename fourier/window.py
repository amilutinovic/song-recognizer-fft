import numpy as np


def hann_window(length):
    """Periodic Hann window of the requested length."""
    if length <= 0:
        raise ValueError("Window length must be positive")

    n = np.arange(length)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)


def hamming_window(length):
    """Periodic Hamming window of the requested length."""
    if length <= 0:
        raise ValueError("Window length must be positive")

    n = np.arange(length)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / length)


def blackman_window(length):
    """Periodic Blackman window."""
    if length <= 0:
        raise ValueError("Window length must be positive")

    n = np.arange(length)
    return (
        0.42
        - 0.5 * np.cos(2.0 * np.pi * n / length)
        + 0.08 * np.cos(4.0 * np.pi * n / length)
    )


def get_window(name, length):
    """Return one of the supported windows by name."""
    name = name.lower()

    if name == "hann":
        return hann_window(length)
    if name == "hamming":
        return hamming_window(length)
    if name == "blackman":
        return blackman_window(length)

    raise ValueError(
        f"Unknown window '{name}'. "
        "Use 'hann', 'hamming' or 'blackman'."
    )

