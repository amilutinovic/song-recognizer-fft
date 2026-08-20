from dataclasses import dataclass

@dataclass(frozen=True)
class Peak:
    time: int
    frequency: int


# Fingerprint generated from a pair of spectral peaks, and their time difference.
@dataclass(frozen=True)
class Fingerprint:
    f1: int
    f2: int
    dt: int