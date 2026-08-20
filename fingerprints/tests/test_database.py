from pathlib import Path

from fingerprint import Peak
from hashing import generate_hashes
from database import FingerprintDatabase


def test_database():

    db_path = Path("data/test_fingerprints.db")

    # Remove old test database so every test starts clean.
    if db_path.exists():
        db_path.unlink()

    db = FingerprintDatabase(db_path)

    peaks = [
        Peak(time=10, frequency=100),
        Peak(time=15, frequency=150),
        Peak(time=20, frequency=200),
        Peak(time=30, frequency=250),
    ]

    fingerprints = generate_hashes(
        peaks,
        fanout=2,
        min_dt=1,
        max_dt=25,
    )

    song_id = db.add_song(
        title="Test Song",
        path="songs/test.wav",
    )

    db.add_fingerprints(
        song_id,
        fingerprints,
    )

    # Check that the song was added.
    song = db.get_song(song_id)

    assert song == (
        song_id,
        "Test Song",
        "songs/test.wav",
    )

    # Check fingerprint lookup.
    fingerprint, anchor_time = fingerprints[0]

    matches = db.lookup(fingerprint)

    assert (song_id, anchor_time) in matches

    db.close()

    print("Database test passed!")


if __name__ == "__main__":
    test_database()