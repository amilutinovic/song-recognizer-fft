import numpy as np
from core import fft_batch, ifft_batch
from window import get_window


def frame_signal(
    signal,
    frame_length,
    hop_length,
):
    """
    Split a 1D signal into overlapping frames.

    The last frame is zero-padded if necessary.

    Returns:
        frames:
            Array of shape (num_frames, frame_length)
        original_length:
            Length of the input signal before padding.
    """
    signal = np.asarray(signal, dtype=float)

    original_length = len(signal)

    if original_length == 0:
        return np.empty((0, frame_length)), 0

    # Number of frames needed to cover the whole signal.
    n_frames = max(
        1,
        int(np.ceil((original_length - frame_length) / hop_length)) + 1,
    )

    padded_length = (n_frames - 1) * hop_length + frame_length

    padded_signal = np.pad(
        signal,
        (0, padded_length - original_length),
        mode="constant",
    )

    frames = np.stack(
        [
            padded_signal[
                start:start + frame_length
            ]
            for start in range(0, n_frames * hop_length, hop_length)
        ]
    )

    return frames, original_length



def stft(
    signal,
    frame_length = 1024,
    hop_length = None,
    window = "hann",
    n_fft = None,
):
    """
    Compute the Short-Time Fourier Transform.

    Returns:
        spectrum:
            Complex array with shape (num_frames, n_fft)

        params:
            Dictionary containing the parameters needed by istft.
    """
    if hop_length is None:
        hop_length = frame_length // 2

    if n_fft is None:
        n_fft = frame_length

    frames, original_length = frame_signal(
        signal,
        frame_length,
        hop_length,
    )

    win = get_window(window, frame_length)

    # Apply the analysis window.
    windowed_frames = frames * win[None, :]

    # Zero-padding in the frequency-transform dimension.
    if n_fft > frame_length:
        padded_frames = np.pad(
            windowed_frames,
            ((0, 0), (0, n_fft - frame_length)),
            mode="constant",
        )
    else:
        padded_frames = windowed_frames

    spectrum = fft_batch(padded_frames)

    params = {
        "frame_length": frame_length,
        "hop_length": hop_length,
        "n_fft": n_fft,
        "window": window,
        "original_length": original_length,
        "num_frames": len(frames),
    }

    return spectrum, params


def istft(
    spectrum,
    params,
):
    """
    Reconstruct a signal from the full complex STFT.
    """
    spectrum = np.asarray(spectrum, dtype=complex)

    if spectrum.ndim != 2:
        raise ValueError("spectrum must have shape (num_frames, n_fft)")

    frame_length = params["frame_length"]
    hop_length = params["hop_length"]
    n_fft = params["n_fft"]
    window_name = params["window"]
    original_length = params["original_length"]

    win = get_window(window_name, frame_length)

    # Convert every spectrum frame back to time domain.
    time_frames = ifft_batch(spectrum).real

    # Remove the zero-padding that was added before the FFT.
    time_frames = time_frames[:, :frame_length]

    padded_length = (
        (len(time_frames) - 1) * hop_length + frame_length
        if len(time_frames) > 0
        else 0
    )

    signal = np.zeros(padded_length, dtype=float)
    window_sum = np.zeros(padded_length, dtype=float)

    # Overlap-add.
    for i, frame in enumerate(time_frames):
        start = i * hop_length
        end = start + frame_length

        signal[start:end] += frame * win

        # Because both use the window
        window_sum[start:end] += win ** 2

    # Normalize where the window contributes.
    nonzero = window_sum > 1e-12
    signal[nonzero] /= window_sum[nonzero]

    # Remove the zero-padding added by frame_signal().
    return signal[:original_length]



