"""
Background worker for recording and recognition.

Recording (a few seconds) and matching take real time. 

Each result dict:
    {"title": str, "artist": str, "score": int, "confidence": float}
"""

import os
import sys

from PyQt6.QtCore import QThread, pyqtSignal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "audio"))
sys.path.insert(0, os.path.join(ROOT, "fourier"))
sys.path.insert(0, os.path.join(ROOT, "fingerprints"))

from audio_io import load_audio, SAMPLE_RATE
from stft import stft
from peaks import find_peaks
from hashing import generate_hashes
from database import FingerprintDatabase
from matching import match_query


def split_title(stored_title):
    """The DB stores 'Artist - Title'. Split it back for display."""
    if " - " in stored_title:
        artist, _, title = stored_title.partition(" - ")
        return artist.strip(), title.strip()
    return "", stored_title.strip()


class RecognizerWorker(QThread):
    status = pyqtSignal(str)
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, db_path, signal=None, file_path=None, parent=None):
        """Recognize an already-recorded `signal`, or load and recognize a
        `file_path`. Recording itself is handled by the window via
        MicRecorder, so the worker only does loading (for files) + matching."""
        super().__init__(parent)
        self.db_path = db_path
        self.signal = signal
        self.file_path = file_path

    def run(self):
        """This method runs on the background thread (started by .start())."""
        try:
            if not os.path.exists(self.db_path):
                self.failed.emit(
                    "No database found. Run build_database.py first.")
                return

            # 1) get the clip: either a file to load, or a signal already
            #    recorded by the window's MicRecorder.
            if self.file_path:
                self.status.emit("Loading audio...")
                signal = load_audio(self.file_path, SAMPLE_RATE)
            else:
                signal = self.signal

            if signal is None or len(signal) == 0:
                self.finished.emit([])
                return

            # 2) fingerprint it (this is the Fourier pipeline)
            self.status.emit("Analyzing...")
            spectrum, _ = stft(signal, frame_length=1024, hop_length=512,
                               window="hann")
            peaks = find_peaks(spectrum)
            hashes = generate_hashes(peaks)

            if not hashes:
                self.finished.emit([])
                return

            # 3) match against the database
            self.status.emit("Matching...")
            db = FingerprintDatabase(self.db_path)
            raw = match_query(hashes, db)          # [(song_id, score), ...]

            # 4) turn the top few into display-ready dicts. Confidence is
            # the winner's share of the total top-3 score -- a simple 0..1
            # number that reads well in the UI.
            top = raw[:3]
            total = sum(score for _, score in top) or 1
            results = []
            for song_id, score in top:
                stored = db.get_song(song_id)[1]
                artist, title = split_title(stored)
                results.append({
                    "title": title,
                    "artist": artist,
                    "score": int(score),
                    "confidence": score / total,
                })
            db.close()

            self.finished.emit(results)

        except Exception as exc:
            self.failed.emit(str(exc))