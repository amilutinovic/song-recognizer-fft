"""
Recognize a song from a short audio clip.

The clip can come from a file (a snippet you cut out, or a whole song) or
from the microphone (play a song out loud and record a few seconds). Both
paths run the same pipeline as build_database.py, then match against the DB:

    clip -> STFT -> peaks -> hashes -> match against database

Run from the project main:
    python recognize.py --file data/samples/clip.wav
    python recognize.py --mic 8
    python recognize.py --file "data/songs/LZYBY - Baby Bird.mp3" --start 30 --duration 8
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "audio"))
sys.path.insert(0, os.path.join(ROOT, "fourier"))
sys.path.insert(0, os.path.join(ROOT, "fingerprints"))

from audio_io import load_audio, record_microphone, SAMPLE_RATE
from stft import stft
from peaks import find_peaks
from hashing import generate_hashes
from database import FingerprintDatabase
from matching import match_query


MIN_SCORE = 5           # ignore anything with fewer aligned hashes than this
MIN_RATIO = 1.5         # winner must beat the runner-up by at least this much


def recognize(signal, db, top_n=3):
    """Fingerprint the clip and return the top_n (song_id, score) matches."""
    spectrum, _ = stft(signal, frame_length=1024, hop_length=512, window="hann")
    peaks = find_peaks(spectrum)
    hashes = generate_hashes(peaks)
    results = match_query(hashes, db)
    return results[:top_n], len(peaks), len(hashes)


def decide(results):
    """
    Turn raw scores into a verdict: a confident match, or 'not sure'.

    We accept the top song only if it has enough aligned hashes AND clearly
    beats the second-best candidate. Otherwise we report low confidence,
    which is better than confidently naming the wrong song.
    """
    if not results:
        return None, 0.0
    top_id, top_score = results[0]
    second = results[1][1] if len(results) > 1 else 0
    ratio = top_score / second if second > 0 else float("inf")

    confident = top_score >= MIN_SCORE and ratio >= MIN_RATIO
    return (top_id if confident else None), ratio


def main():
    parser = argparse.ArgumentParser(description="Recognize a song from a clip")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="path to an audio clip or a full song")
    source.add_argument("--mic", type=float, metavar="SECONDS",
                        help="record this many seconds from the microphone")
    parser.add_argument("--start", type=float, default=0.0,
                        help="with --file: start of the clip in seconds")
    parser.add_argument("--duration", type=float, default=None,
                        help="with --file: clip length in seconds "
                             "(default: use the whole file)")
    parser.add_argument("--db", default=os.path.join(ROOT, "data", "fingerprints.db"))
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}\n"
              f"Run 'python build_database.py' first.")
        return

    # Get the clip
    if args.file:
        clip_path = args.file
        if not os.path.exists(clip_path):
            fallback = os.path.join(ROOT, args.file)
            if os.path.exists(fallback):
                clip_path = fallback
            else:
                print(f"File not found: {args.file}\n"
                      f"(also tried: {fallback})")
                return
        signal = load_audio(clip_path, SAMPLE_RATE)
        start = int(args.start * SAMPLE_RATE)
        if args.duration is not None:
            end = start + int(args.duration * SAMPLE_RATE)
            signal = signal[start:end]
        else:
            signal = signal[start:]
        print(f"Clip: {len(signal)/SAMPLE_RATE:.1f}s from {args.file}")
    else:
        print(f"Recording {args.mic:.0f}s from the microphone...")
        signal = record_microphone(args.mic, SAMPLE_RATE)
        print("Recording done.")

    # Recognize
    db = FingerprintDatabase(args.db)
    t0 = time.perf_counter()
    results, num_peaks, num_hashes = recognize(signal, db)
    elapsed = time.perf_counter() - t0

    print(f"\n{num_peaks} peaks, {num_hashes} hashes, matched in {elapsed:.2f}s\n")

    if not results:
        print("No match found.")
        db.close()
        return

    print("Top candidates:")
    for rank, (song_id, score) in enumerate(results, 1):
        title = db.get_song(song_id)[1]
        print(f"  {rank}. {title:40s}  score={score}")

    match_id, ratio = decide(results)
    print()
    if match_id is not None:
        title = db.get_song(match_id)[1]
        print(f"MATCH: {title}   (score {results[0][1]}, "
              f"{ratio:.1f}x the runner-up)")
    else:
        print("Low confidence -- no clear match. "
              "Try a longer or cleaner clip.")

    db.close()


if __name__ == "__main__":
    main()