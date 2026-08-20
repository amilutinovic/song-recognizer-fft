"""
Peak detection in a spectrogram (the constellation map).

From the full spectrogram we keep only the loudest points in their local
neighborhood. These local maxima survive noise, compression and bad room
acoustics, while the rest of the spectrum changes. The result is a sparse
set of (time, frequency) points - the song's "starry sky".

The output is a list of Peak(time, frequency) objects, exactly what
hashing.generate_hashes expects.
"""

import numpy as np
from scipy.ndimage import maximum_filter

from fingerprint import Peak


def spectrogram_db(spectrum):
    """
    Turn the complex STFT into a magnitude spectrogram in decibels.

    Input:  spectrum from stft(), shape (num_frames, n_fft), complex.
    Output: real array (num_frames, n_freq), where n_freq = n_fft//2 + 1,
            values in dB with the maximum normalized to 0.

    
    1. We keep only the first half of the frequency axis. The signal is
    real, so its spectrum is mirror-symmetric; the second half is a
    redundant copy and would just create duplicate peaks.

    2. Magnitude -> dB (20*log10). The ear hears loudness roughly
    logarithmically, and dB makes a fixed threshold meaningful
    across quiet and loud songs.
    """

    spectrum = np.asarray(spectrum)
    num_frames, n_fft = spectrum.shape
    n_freq = n_fft // 2 + 1

    magnitude = np.abs(spectrum[:, :n_freq])          # keep the lower half
    magnitude = np.maximum(magnitude, 1e-10)          # avoid log(0)
    db = 20.0 * np.log10(magnitude)
    db -= db.max()                                    # normalize peak to 0 dB
    return db


def find_peaks(spectrum, neighborhood_time=10, neighborhood_freq=10,
               min_db=-55.0):
    """
    Find local maxima of the spectrogram and return them as 
    list of Peaks sorted by time.

    Parameters:
        spectrum : complex STFT from stft(), shape (num_frames, n_fft)
        neighborhood_time : half-window in frames for the local-max search
        neighborhood_freq : half-window in frequency bins
        min_db : ignore anything quieter than this (relative to the 0 dB peak)

    """
    db = spectrogram_db(spectrum)

    # size = full window (2*half + 1) along each axis
    size = (2 * neighborhood_time + 1, 2 * neighborhood_freq + 1)
    local_max = maximum_filter(db, size=size, mode="constant", cval=-np.inf)

    # A point is a peak if it equals the local maximum AND is loud enough.
    is_peak = (db == local_max) & (db >= min_db)

    frames, freqs = np.nonzero(is_peak)               # row=time, col=freq
    peaks = [Peak(time=int(t), frequency=int(f))
             for t, f in zip(frames, freqs)]

    # Sort by time so hashing can slide forward through targets.
    peaks.sort(key=lambda p: p.time)
    return peaks


def peak_density(peaks, num_frames, frames_per_second):
    """Average peaks per second. Use it to sanity-check the threshold:
    too few peaks -> misses, too many -> a huge database and false hits.
    Aim for very roughly 20-40 peaks per second."""
    if num_frames == 0:
        return 0.0
    return len(peaks) / (num_frames / frames_per_second)