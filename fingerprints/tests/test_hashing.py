from fingerprint import Peak
from hashing import generate_hashes


def test_generate_hashes():
    peaks = [
        Peak(time=10, frequency=100),
        Peak(time=15, frequency=150),
        Peak(time=20, frequency=200),
        Peak(time=30, frequency=250),
    ]

    result = generate_hashes(
        peaks,
        fanout=2,
        min_dt=1,
        max_dt=25,
    )

    for fingerprint, anchor_time in result:
        print(
            f"fingerprint={fingerprint}, "
            f"anchor_time={anchor_time}"
        )


if __name__ == "__main__":
    test_generate_hashes()