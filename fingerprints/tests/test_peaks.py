import os
import sys

import numpy as np

# Make fingerprints/ and fourier/ importable regardless of run location.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "fingerprints"))
sys.path.insert(0, os.path.join(ROOT, "fourier"))

from peaks import find_peaks, spectrogram_db, peak_density
from fingerprint import Peak
from stft import stft

SAMPLE_RATE = 11025

def make_tone_signal(freqs, seconds=3.0, fs=SAMPLE_RATE):
    """A signal built from a few pure tones at known frequencies."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    signal = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return signal / np.max(np.abs(signal))


def test_spectrogram_db_shape_and_scale():
    """dB spectrogram keeps the lower half of frequencies and peaks at 0."""
    signal = make_tone_signal([440])
    spectrum, _ = stft(signal, frame_length=1024, hop_length=512)
    db = spectrogram_db(spectrum)
    num_frames, n_fft = spectrum.shape
    assert db.shape == (num_frames, n_fft // 2 + 1)
    assert np.isclose(db.max(), 0.0)          # normalized to 0 dB
    assert np.all(db <= 0.0)                  # everything else is quieter


def test_find_peaks_returns_peak_objects():
    """Output must be Peak objects (so hashing can consume them)."""
    signal = make_tone_signal([440, 1000])
    spectrum, _ = stft(signal, frame_length=1024, hop_length=512)
    peaks = find_peaks(spectrum)
    assert len(peaks) > 0
    assert all(isinstance(p, Peak) for p in peaks)
    assert all(hasattr(p, "time") and hasattr(p, "frequency") for p in peaks)


def test_find_peaks_finds_the_right_frequencies():
    """The detected frequency bins should match the input tones.
    bin -> Hz conversion: f = bin * fs / n_fft."""
    fs, n_fft = SAMPLE_RATE, 1024
    signal = make_tone_signal([440, 1000], fs=fs)
    spectrum, _ = stft(signal, frame_length=n_fft, hop_length=512)
    peaks = find_peaks(spectrum)

    detected_hz = {round(p.frequency * fs / n_fft) for p in peaks}
    # allow +/- one bin of tolerance
    for target in (440, 1000):
        assert any(abs(hz - target) <= fs / n_fft for hz in detected_hz), \
            f"missing tone near {target} Hz, got {sorted(detected_hz)}"


def test_peaks_sorted_by_time():
    signal = make_tone_signal([300, 600, 900])
    spectrum, _ = stft(signal, frame_length=1024, hop_length=512)
    peaks = find_peaks(spectrum)
    times = [p.time for p in peaks]
    assert times == sorted(times)


def test_threshold_removes_quiet_points():
    """A stricter (higher) threshold keeps fewer peaks."""
    signal = make_tone_signal([440, 1000], seconds=5.0)
    spectrum, _ = stft(signal, frame_length=1024, hop_length=512)
    many = find_peaks(spectrum, min_db=-80)
    few = find_peaks(spectrum, min_db=-20)
    assert len(few) <= len(many)


def test_peak_density_matches_manual():
    peaks = [Peak(time=i, frequency=10) for i in range(50)]
    # 50 peaks over 100 frames at 20 frames/sec -> 5 seconds -> 10 peaks/sec
    assert np.isclose(peak_density(peaks, num_frames=100, frames_per_second=20), 10.0)


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
        print("All peak-detection tests passed.")