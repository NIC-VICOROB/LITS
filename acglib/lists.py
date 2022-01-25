import numpy as np

def resample_regular(l, n):
    """Resamples a given list to have length `n`.

    List elements are repeated or removed at regular intervals to reach the desired length.

    :param list l: list to resample
    :param int n: desired length of resampled list
    :return list: the resampled list of length `n`

    :Example:

    >>> resample_regular([0, 1, 2, 3, 4, 5], n=3)
    [0, 2, 4]
    >>> resample_regular([0, 1, 2, 3], n=6)
    [0, 1, 2, 3, 0, 2]
    """
    n = int(n)
    if n <= 0:
        return []

    if len(l) < n:  # List smaller than n (Repeat elements)
        resampling_idxs = list(range(len(l))) * (n // len(l))  # Full repetitions

        if len(resampling_idxs) < n:  # Partial repetitions
            resampling_idxs += np.round(np.arange(
                start=0., stop=float(len(l)) - 1., step=len(l) / float(n % len(l))), decimals=0).astype(int).tolist()

        assert len(resampling_idxs) == n
        return [l[i] for i in resampling_idxs]
    elif len(l) > n:  # List bigger than n (Subsample elements)
        resampling_idxs = np.round(np.arange(
            start=0., stop=float(len(l)) - 1., step=len(l) / float(n)), decimals=0).astype(int).tolist()

        assert len(resampling_idxs) == n
        return [l[i] for i in resampling_idxs]
    else:
        return l