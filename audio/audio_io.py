"""
Audio loading, preprocessing and recording.

Turns any audio file (or a microphone recording) into the clean 1D signal
the rest of the pipeline expects: mono, a fixed sample rate, float values
in [-1, 1]. librosa/sounddevice are used only as decoders/recorders --
no Fourier processing happens here.
"""

import numpy as np
import librosa
import sounddevice as sd

# These parameters are shared with the STFT layer. If your colleague later.
SAMPLE_RATE = 11025      # Hz. Enough for music (most energy is below 5 kHz)
                         # and 4x cheaper than 44.1 kHz.


def normalize(signal):
    """
    Scale the signal so its loudest sample is +/-1.

    A quiet phone recording and a loud studio file of the same song must
    produce the same spectrogram shape, so we remove the overall volume
    here. Silence (all zeros) is returned unchanged to avoid dividing by 0.
    """
    signal = np.asarray(signal, dtype=np.float32)
    peak = np.max(np.abs(signal))
    if peak < 1e-9:
        return signal
    return signal / peak


def load_audio(path, sample_rate=SAMPLE_RATE):
    """
    Load an audio file into a mono float32 signal at sample_rate.

    Steps: decode -> average channels to mono -> resample to sample_rate
    -> normalize.
    Supports every format ffmpeg can read (mp3, wav, flac, m4a, ogg). 
    librosa does the decode/mono/resample; we only normalize.
    """

    signal, _ = librosa.load(path, sr=sample_rate, mono=True)
    return normalize(signal.astype(np.float32))


def record_microphone(seconds, sample_rate=SAMPLE_RATE):
    """
    Record from the default microphone and return a mono float32 signal.

    Used by the "listen and recognize" mode. Blocks until recording ends.
    """
    num_samples = int(seconds * sample_rate)
    recording = sd.rec(num_samples, samplerate=sample_rate,
                       channels=1, dtype="float32")
    sd.wait()                      # block until the recording is finished
    return normalize(recording.flatten())


def add_noise(signal, snr_db, seed=None):
    """
    Add white Gaussian noise at a given signal-to-noise ratio (dB).

    Used later in the robustness evaluation, to test how well recognition
    survives noise. Higher snr_db = less noise.
        SNR_dB = 10 * log10(signal_power / noise_power)
    """
    rng = np.random.default_rng(seed)
    signal = np.asarray(signal, dtype=np.float32)
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return normalize(signal + noise.astype(np.float32))