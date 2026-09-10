"""
Experiment: recognition WITH vs WITHOUT a window function.

Builds two databases from the same songs - one with the Hann window, one
with a rectangular window (i.e. no window) - then recognizes clips of each
song against its matching database, both clean and with added noise. Prints
a table of scores so we can see whether the window matters for recognition.

Both the database and the query use the same window in each column, so the
comparison is fair (base and query always processed identically).

Run from the project root:
    python compare_windows.py
    python compare_windows.py --snr 5 --duration 8
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("audio", "fourier", "fingerprints"):
    sys.path.insert(0, os.path.join(ROOT, sub))

from audio_io import load_audio, add_noise, SAMPLE_RATE
from stft import stft
from peaks import find_peaks
from hashing import generate_hashes
from database import FingerprintDatabase
from matching import match_query

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def fingerprint(signal, window):
    """Full offline pipeline for one signal, with the chosen window."""
    spectrum, _ = stft(signal, frame_length=1024, hop_length=512, window=window)
    return generate_hashes(find_peaks(spectrum))


def build_db(songs_dir, window, db_path):
    """Build a fresh database from all songs, using the given window."""
    if os.path.exists(db_path):
        os.remove(db_path)
    db = FingerprintDatabase(db_path)
    titles = {}
    for fname in sorted(os.listdir(songs_dir)):
        if os.path.splitext(fname)[1].lower() not in AUDIO_EXTENSIONS:
            continue
        signal = load_audio(os.path.join(songs_dir, fname))
        title = os.path.splitext(fname)[0]
        sid = db.add_song_with_fingerprints(title, fingerprint(signal, window),
                                            path=fname)
        titles[sid] = title
    return db, titles


def recognize(signal, db, window):
    """Return (winner_song_id, winner_score) for a signal, or (None, 0)."""
    results = match_query(fingerprint(signal, window), db)
    if not results:
        return None, 0
    return results[0]


def main():
    parser = argparse.ArgumentParser(description="Compare window vs no window")
    parser.add_argument("--songs", default=os.path.join(ROOT, "data", "songs"))
    parser.add_argument("--start", type=float, default=10.0,
                        help="clip start in seconds")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="clip length in seconds")
    parser.add_argument("--snr", type=float, default=5.0,
                        help="signal-to-noise ratio (dB) for the noisy test")
    args = parser.parse_args()

    # Build both databases once.
    print("Building databases...")
    db_hann, titles = build_db(args.songs, "hann",
                               os.path.join(ROOT, "data", "fp_hann.db"))
    db_rect, _ = build_db(args.songs, "rect",
                          os.path.join(ROOT, "data", "fp_rect.db"))

    files = sorted(f for f in os.listdir(args.songs)
                   if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS)

    print(f"\nClip: {args.duration:.0f}s from {args.start:.0f}s, "
          f"noisy test at SNR {args.snr:.0f} dB\n")
    print(f"{'song':22} {'hann clean':>11} {'rect clean':>11} "
          f"{'hann noisy':>11} {'rect noisy':>11}")
    print("-" * 70)

    hann_ok_clean = rect_ok_clean = hann_ok_noisy = rect_ok_noisy = 0

    for fname in files:
        title = os.path.splitext(fname)[0]
        signal = load_audio(os.path.join(args.songs, fname))
        a = int(args.start * SAMPLE_RATE)
        b = a + int(args.duration * SAMPLE_RATE)
        clip = signal[a:b]
        noisy = add_noise(clip, args.snr, seed=0)

        # recognize the clean clip and the noisy clip against each database
        h_clean = recognize(clip, db_hann, "hann")
        r_clean = recognize(clip, db_rect, "rect")
        h_noisy = recognize(noisy, db_hann, "hann")
        r_noisy = recognize(noisy, db_rect, "rect")

        # a hit means the top match is the same song we clipped from
        def hit(res):
            sid = res[0]
            return sid is not None and titles.get(sid) == title

        hann_ok_clean += hit(h_clean)
        rect_ok_clean += hit(r_clean)
        hann_ok_noisy += hit(h_noisy)
        rect_ok_noisy += hit(r_noisy)

        def cell(res):
            mark = "OK" if hit(res) else "X"
            return f"{res[1]:>7} {mark:>3}"

        print(f"{title[:22]:22} {cell(h_clean)} {cell(r_clean)} "
              f"{cell(h_noisy)} {cell(r_noisy)}")

    n = len(files)
    print("-" * 70)
    print(f"{'correct:':22} {hann_ok_clean:>7}/{n:<3} {rect_ok_clean:>7}/{n:<3} "
          f"{hann_ok_noisy:>7}/{n:<3} {rect_ok_noisy:>7}/{n:<3}")

    db_hann.close()
    db_rect.close()

    print("\nEach cell shows the winning score and whether it was correct.")
    print("Compare 'hann noisy' vs 'rect noisy' - that is where the window")
    print("is expected to help, because leakage hurts most under noise.")


if __name__ == "__main__":
    main()