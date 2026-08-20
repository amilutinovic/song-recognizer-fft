from fingerprint import Peak, Fingerprint

def generate_hashes(
        peaks: list[Peak],
        fanout: int = 10,
        min_dt: int = 1,
        max_dt: int = 100
):
    """
    Generate fingerprints from a constellation map.

    For every anchor peak, pair it with up to fanout
    following peaks whose time difference is within
    [min_dt, max_dt].

    Returns:
        A list of (Fingerprint, anchor_time) pairs.
    """

    # Sort peaks by time so that "following peaks"
    # are processed in chronological order.
    peaks = sorted(peaks, key=lambda peak: peak.time)

    fingerprints = []

    for i, anchor in enumerate(peaks):

        targets_used = 0

        for target in peaks[i + 1:]:

            dt = target.time - anchor.time

            # Since peaks are sorted by time, once dt is too large,
            # all following peaks will also be too far away.
            if dt > max_dt:
                break

            if dt < min_dt:
                continue

            fingerprint = Fingerprint(
                f1=anchor.frequency,
                f2=target.frequency,
                dt=dt,
            )

            fingerprints.append(
                (fingerprint, anchor.time)
            )

            targets_used += 1

            if targets_used >= fanout:
                break

    return fingerprints