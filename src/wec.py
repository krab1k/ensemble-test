import Bio.PDB
import numpy as np
import argparse
import itertools
import sys
from sklearn import manifold, decomposition
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

import simplex


def get_distance_matrix(ensemble1, ensemble2):
    all_structures = list(ensemble1) + list(ensemble2)
    n = len(all_structures)
    distance_matrix = np.zeros((n, n), dtype=np.float_)
    for i, j in itertools.combinations(range(n), 2):
        superimposer = Bio.PDB.Superimposer()
        superimposer.set_atoms(list(all_structures[i].get_atoms()), list(all_structures[j].get_atoms()))
        distance = superimposer.rms
        distance_matrix[i, j] = distance_matrix[j, i] = distance

    return distance_matrix


def kullback_leibler_divergence(p_x, p_y):
    return np.average(p_x * np.log(p_x / p_y))


def jensen_shannon_divergence(kernel1, kernel2, grid):
    p_x = kernel1(grid)
    p_y = kernel2(grid)
    m = 0.5 * (p_x + p_y)
    return 0.5 * (kullback_leibler_divergence(p_x, m) + kullback_leibler_divergence(p_y, m))


def load_weights(filename):
    try:
        with open(filename) as f:
            weights = [float(v) for v in (f.readline() + f.readline()).split()]
            return weights
    except IOError:
        print(f'Cannot open file with weights {filename}', file=sys.stderr)
        sys.exit(1)


def adjust_data_by_weights(data, weights, size1):
    tmp = []
    for d, w in zip(data, weights):
        tile = np.tile(d, (int(w * 100), 1))
        tmp.append(tile)

    res = np.vstack(tmp)
    return res, sum(int(w * 100) for w in weights[:size1]), sum(int(w * 100) for w in weights[size1:])


def plot_1d(kernel1, kernel2, data1, data2, grid):
    plt.plot(grid, kernel1.evaluate(grid), 'r')
    plt.plot(grid, kernel2.evaluate(grid), 'b')

    plt.scatter(data1, np.full(len(data1), -0.01), c='r', marker='.', alpha=0.5)
    plt.scatter(data2, np.full(len(data2), -0.02), c='b', marker='.', alpha=0.5)
    plt.show()


def plot_2d(kernel1, kernel2, data1, data2, grid):
    res1 = kernel1.evaluate(grid)
    res2 = kernel2.evaluate(grid)

    z1 = np.reshape(res1.T, xx.shape)
    z2 = np.reshape(res2.T, xx.shape)

    fig, ax = plt.subplots()
    ax.imshow(np.rot90(z1), cmap=plt.cm.Reds, extent=[xmin, xmax, ymin, ymax], alpha=0.5)
    ax.imshow(np.rot90(z2), cmap=plt.cm.Blues, extent=[xmin, xmax, ymin, ymax], alpha=0.5)

    plt.scatter(*data1, c='r', alpha=0.5)
    plt.scatter(*data2, c='b', alpha=0.5)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ensemble1')
    parser.add_argument('ensemble2')
    parser.add_argument('-d', '--dims', default=2, choices=[1, 2], type=int)
    parser.add_argument('-w', '--weights', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')

    args = parser.parse_args()

    try:
        pdb_parser = Bio.PDB.PDBParser(QUIET=True)
        ensemble1 = pdb_parser.get_structure('en1', args.ensemble1)
        ensemble2 = pdb_parser.get_structure('en2', args.ensemble2)
    except IOError:
        print(f'Cannot load ensemble files: {args.ensemble1}, {args.ensemble2}', file=sys.stderr)
        sys.exit(1)

    ensemble1_size = len(ensemble1)
    ensemble2_size = len(ensemble2)

    if args.verbose:
        print(f'Loaded ensemble 1 ({ensemble1_size} models), ensemble 2 ({ensemble2_size} models)')

    distance_matrix = get_distance_matrix(ensemble1, ensemble2)
    np.set_printoptions(suppress=True)

    if not simplex.check_triangle_inequality(distance_matrix):
        print('Triangle inequality doesn\'t hold for the distance matrix, something bad had happed.', file=sys.stderr)
        sys.exit(1)

    coords = simplex.n_simplex_build(distance_matrix, check_result=True)
    pca = decomposition.PCA(n_components=args.dims)
    transformed = pca.fit_transform(coords)

    if args.verbose:
        print(f'Performed PCA from {coords.shape[1]} to {args.dims} dimension(s)')

    if args.weights:
        weights = load_weights(args.weights)
        if args.verbose:
            print('Loaded weights:')
            print(f'ensemble 1: {weights[:ensemble1_size]}')
            print(f'ensemble 2: {weights[ensemble1_size:]}')
        transformed, ensemble1_size, ensemble2_size = adjust_data_by_weights(transformed, weights, ensemble1_size)

    print(transformed)
    if args.dims == 1:
        ensemble1_data = transformed[:ensemble1_size].T[0]
        ensemble2_data = transformed[ensemble1_size:].T[0]

        kernel1 = gaussian_kde(ensemble1_data)
        kernel2 = gaussian_kde(ensemble2_data)

        xmin, xmax = np.min(transformed), np.max(transformed)
        grid = np.linspace(xmin, xmax, 100)

        if args.plot:
            plot_1d(kernel1, kernel2, ensemble1_data, ensemble2_data, grid)
    else:
        ensemble1_data = transformed[:ensemble1_size, :].T
        ensemble2_data = transformed[ensemble1_size:, :].T

        kernel1 = gaussian_kde(ensemble1_data)
        kernel2 = gaussian_kde(ensemble2_data)

        xmin, ymin = np.min(transformed, 0)
        xmax, ymax = np.max(transformed, 0)
        xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        grid = np.vstack([xx.ravel(), yy.ravel()])

        if args.plot:
            plot_2d(kernel1, kernel2, ensemble1_data, ensemble2_data, grid)

    print('Final divergence = {:f}'.format(jensen_shannon_divergence(kernel1, kernel2, grid)))


if __name__ == '__main__':
    main()
