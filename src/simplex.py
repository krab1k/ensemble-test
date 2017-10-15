import itertools
import numpy as np


def check_triangle_inequality(data) -> bool:
    n = data.shape[0]

    return all(data[i, j] + data[j, k] >= data[i, k]
               for i, j, k in itertools.combinations(range(n), 3))


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return np.linalg.norm(v1 - v2)


def check_final_distances(data: np.ndarray, simplex: np.ndarray) -> bool:
    n = simplex.shape[0]
    results = []

    tol = 1e-6
    for i, j in itertools.combinations_with_replacement(range(n), 2):
        d_new = euclidean_distance(simplex[i], simplex[j])
        d_orig = data[i, j]
        results.append(abs(d_new - d_orig) < tol)

    return all(results)


def apex_addition(sigma_base: np.ndarray, distances: np.ndarray) -> np.ndarray:
    n = sigma_base.shape[0]
    output = np.zeros(n)
    output[0] = distances[0]

    for i in range(1, n):
        dist = euclidean_distance(np.concatenate((sigma_base[i], np.float_([0]))), output)
        delta = distances[i]
        x = sigma_base[i][i - 1]
        y = output[i - 1]
        output[i - 1] = y - (delta ** 2 - dist ** 2) / (2 * x)
        output[i] = np.sqrt(y ** 2 - output[i - 1] ** 2)

    return output


def n_simplex_build(data: np.ndarray, check_result=False) -> np.ndarray:
    n = data.shape[0] - 1
    sigma = np.zeros((n + 1, n))

    if n == 1:
        distance = data[0, 1]
        sigma[:] = np.array([[0], [distance]])
        return sigma

    sigma_base = n_simplex_build(data[:-1, :-1])
    distances = np.zeros(n)
    for i in range(n):
        distances[i] = data[i, n]

    new_apex = apex_addition(sigma_base, distances)

    for i in range(n):
        for j in range(i):
            sigma[i, j] = sigma_base[i, j]

    for j in range(n):
        sigma[n][j] = new_apex[j]

    if check_result and not check_final_distances(data, sigma):
        print('WARNING: Simplex probably incorrect!')

    return sigma
