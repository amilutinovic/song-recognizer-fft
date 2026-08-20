from collections import defaultdict


# Offsets that are close enough will belong to the same bin.
def get_offset_bin(
    offset,
    tolerance
):

    return round(offset / tolerance)


def match_query(
    query_fingerprints,
    database,
    offset_tolerance = 2,
):

    # song_id -> offset -> number of matches
    offset_counts = defaultdict(lambda: defaultdict(int))

    for fingerprint, query_time in query_fingerprints:

        matches = database.lookup(fingerprint)

        for song_id, database_time in matches:

            offset = database_time - query_time

            offset_bin = round(offset / offset_tolerance)

            offset_counts[song_id][offset_bin] += 1

    results = []

    for song_id, offsets in offset_counts.items():

        score = max(offsets.values())

        results.append(
            (song_id, score)
        )

    results.sort(
        key=lambda result: result[1],
        reverse=True,
    )

    return results