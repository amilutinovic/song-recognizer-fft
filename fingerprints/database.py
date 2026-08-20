import sqlite3
from pathlib import Path

from fingerprint import Fingerprint


class FingerprintDatabase:
    """
    The database contains two tables:

        songs:
            Stores information about each song.

        fingerprints:
            Stores each fingerprint together with the song
            and the time at which the fingerprint occurs.
    """

    def __init__(self, db_path):
        self.db_path = Path(db_path)

        # Create parent directory if it does not exist.
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.db_path)

        self._create_tables()

    def _create_tables(self):
        """Create database tables and indexes."""

        cursor = self.connection.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                path TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fingerprints (
                f1 INTEGER NOT NULL,
                f2 INTEGER NOT NULL,
                dt INTEGER NOT NULL,
                song_id INTEGER NOT NULL,
                time INTEGER NOT NULL,

                FOREIGN KEY (song_id)
                    REFERENCES songs(id)
            )
        """)

        # The matching algorithm will frequently search
        # for fingerprints using f1, f2 and dt.
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_fingerprint_hash
            ON fingerprints(f1, f2, dt)
        """)

        self.connection.commit()

    def add_song(
        self,
        title,
        path= None,
    ):
        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO songs (title, path)
            VALUES (?, ?)
            """,
            (title, path),
        )

        self.connection.commit()

        return cursor.lastrowid

    def add_fingerprints(
        self,
        song_id,
        fingerprints
    ):
        rows = [
            (
                fingerprint.f1,
                fingerprint.f2,
                fingerprint.dt,
                song_id,
                anchor_time,
            )
            for fingerprint, anchor_time in fingerprints
        ]

        if not rows:
            return

        cursor = self.connection.cursor()

        cursor.executemany(
            """
            INSERT INTO fingerprints
                (f1, f2, dt, song_id, time)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )

        self.connection.commit()

    def add_song_with_fingerprints(
        self,
        title,
        fingerprints,
        path = None,
    ):
        song_id = self.add_song(title, path)

        self.add_fingerprints(
            song_id,
            fingerprints,
        )

        return song_id

    def lookup(
        self,
        fingerprint
    ):
        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT song_id, time
            FROM fingerprints
            WHERE f1 = ?
              AND f2 = ?
              AND dt = ?
            """,
            (
                fingerprint.f1,
                fingerprint.f2,
                fingerprint.dt,
            ),
        )

        return cursor.fetchall()

    def get_song(self, song_id: int):
        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT id, title, path
            FROM songs
            WHERE id = ?
            """,
            (song_id,),
        )

        return cursor.fetchone()

    def get_all_songs(self):

        cursor = self.connection.cursor()

        cursor.execute("""
            SELECT id, title, path
            FROM songs
            ORDER BY id
        """)

        return cursor.fetchall()

    def close(self):
        self.connection.close()