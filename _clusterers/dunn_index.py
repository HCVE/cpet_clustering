import numpy as np

"""This module calculates the dunn index as implemented by Joaquim L. Viegas 2016. 
https://github.com/jqmviegas/jqm_cvi. 
The module is modified so that a distance matrix (instead of the training data) can be used directly.
"""


def delta_fast(ck, cl, distances):
    values = distances[np.where(ck)][:, np.where(cl)]
    values = values[np.nonzero(values)]

    return np.min(values)


def big_delta_fast(ci, distances):
    values = distances[np.where(ci)][:, np.where(ci)]
    return np.max(values)


def dunn_fast(distances, labels):
    """ Dunn index - FAST

    Parameters
    ----------
    distances : distance matrix
    labels: np.array
        np.array([N]) labels of all points
    """
    ks = np.sort(np.unique(labels))

    deltas = np.ones([len(ks), len(ks)]) * 1000000
    big_deltas = np.zeros([len(ks), 1])

    l_range = list(range(0, len(ks)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = delta_fast((labels == ks[k]), (labels == ks[l]), distances)

        big_deltas[k] = big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas) / np.max(big_deltas)
    return di
