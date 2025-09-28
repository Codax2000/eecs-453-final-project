'''
Alex Knowlton
12/7/2024

This file presents the analytical solution of the problem defined in the
project proposal
'''

import numpy as np


def analytical_solve(vector, constellations):
    '''
    Given a signal vector and a list of constellations, returns the index of
    the constellation with the lowest mean Euclidean distance from the given
    vector.
    '''
    results = []
    for constellation in constellations:
        dE = get_mean_euclidean_distance(vector, constellation)
        results.append(dE)
    results = np.array(results)
    return np.argmin(results)


def get_mean_euclidean_distance(vector, constellation):
    '''
    Given a complex signal vector and a constellation vector, computes the most
    likely index within the constellation and computes the euclidean distance
    from that point in the signal vector and that point in the constellation,
    and returns the mean of all the euclidean distances.
    '''
    vector = vector.T.reshape((2, 128)).T
    vector = vector[:, 0] + 1j * vector[:, 1]
    vector = vector.reshape((vector.shape[0], 1))
    constellation = constellation.T
    diff = vector - constellation
    diff = np.abs(diff * np.conj(diff))  # squared euclidean distance
    min = np.argmin(diff, axis=1)
    selected = constellation[:, min].T
    diff = vector - selected
    diff = np.abs(diff * np.conj(diff))
    return np.mean(diff)
