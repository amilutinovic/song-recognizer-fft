"""
Build the fingerprint database from every song in data/songs/.

Runs the full offline pipeline on each file:
    audio -> STFT -> peaks -> hashes -> database

File naming: "Artist - Title.mp3". If a file has no " - " separator, the
whole name becomes the title and the artist is left as "Unknown".
"""

import argparse
import os
import sys
import time

# Make the flat packages importable (audio/, fourier/, fingerprints/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "audio"))
sys.path.insert(0, os.path.join(ROOT, "fourier"))
sys.path.insert(0, os.path.join(ROOT, "fingerprints"))

from audio_io import load_audio, SAMPLE_RATE
from stft import stft
from peaks import find_peaks
from hashing import generate_hashes
from database import FingerprintDatabase

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def split_artist_title(filename):
    """Turn a file name into (artist, title).

    Accepts 'Artist - Title', 'Artist-Title' and 'Artist_-_Title', and
    tidies stray underscores so metadata stays readable no matter how the
    file was named on download.
    """
    name = os.path.splitext(filename)[0]

    # Try the clean separator first, then messier variants.
    for sep in (" - ", "_-_", " -", "- ", "-"):
        if sep in name:
            artist, _, title = name.partition(sep)
            artist = artist.replace("_", " ").strip()
            title = title.replace("_", " ").strip()
            if artist and title:
                return artist, title

    return "Unknown", name.replace("_", " ").strip()


def process_song(path):
    """Run one song through the whole offline pipeline and return its
    (signal_seconds, num_peaks, hashes)."""
    signal = load_audio(path, SAMPLE_RATE)
    spectrum, _ = stft(signal, frame_length=1024, hop_length=512, window="hann")
    peaks = find_peaks(spectrum)
    hashes = generate_hashes(peaks)
    return len(signal) / SAMPLE_RATE, len(peaks), hashes


def main():
    parser = argparse.ArgumentParser(description="Build the fingerprint DB")
    parser.add_argument("--songs", default=os.path.join(ROOT, "data", "songs"))
    parser.add_argument("--db", default=os.path.join(ROOT, "data", "fingerprints.db"))
    args = parser.parse_args()

    if not os.path.isdir(args.songs):
        print(f"Songs folder not found: {args.songs}")
        return

    files = sorted(f for f in os.listdir(args.songs)
                   if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS)
    if not files:
        print(f"No audio files in {args.songs}")
        return

    db = FingerprintDatabase(args.db)
    print(f"Building database from {len(files)} songs -> {args.db}\n")

    total_hashes = 0
    start = time.perf_counter()

    for i, filename in enumerate(files, 1):
        path = os.path.join(args.songs, filename)
        artist, title = split_artist_title(filename)
        t0 = time.perf_counter()
        try:
            duration, num_peaks, hashes = process_song(path)
        except Exception as exc:
            print(f"[{i}/{len(files)}] ERROR {filename}: {exc}")
            continue

        db.add_song_with_fingerprints(f"{artist} - {title}", hashes, path=path)
        total_hashes += len(hashes)
        print(f"[{i}/{len(files)}] {artist} - {title}")
        print(f"        {duration:5.1f}s  {num_peaks:5d} peaks  "
              f"{len(hashes):6d} hashes  ({time.perf_counter()-t0:.2f}s)")

    elapsed = time.perf_counter() - start
    print(f"\nDone: {len(files)} songs, {total_hashes} hashes in {elapsed:.1f}s")
    db.close()


if __name__ == "__main__":
    main()