import Bio.PDB
import numpy as np
import argparse
import itertools
import sys
from sklearn import manifold, decomposition
from scipy.stats import gaussian_kde, entropy
import matplotlib.pyplot as plt

import simplex


def get_distance(coords1, coords2):
    superimposer = Bio.PDB.Superimposer()
    superimposer.set_atoms(coords1, coords2)
    return superimposer.rms


def get_distance_matrix(ensemble1, ensemble2):
    all_atoms = [list(structure.get_atoms()) for structure in itertools.chain(ensemble1, ensemble2)]
    n = len(all_atoms)

    distance_matrix = np.zeros((n, n), dtype=np.float_)
    for i, j in itertools.combinations(range(n), 2):
        distance_matrix[i, j] = distance_matrix[j, i] = get_distance(all_atoms[i], all_atoms[j])

    return distance_matrix


def jensen_shannon_distance(kernel1, kernel2, grid):
    p_x = kernel1(grid)
    p_y = kernel2(grid)
    m = 0.5 * (p_x + p_y)
    return (0.5 * (entropy(p_x, m) + entropy(p_y, m))) ** 0.5


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


def plot_2d(kernel1, kernel2, data1, data2, grid, xx, xmin, xmax, ymin, ymax):
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
    parser.add_argument('--distance-matrix', type=str)

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

    if args.distance_matrix:
        try:
            with open(args.distance_matrix, 'rb') as f:
                distance_matrix = np.load(f)
            print('Loaded distance matrix')
        except IOError:
            distance_matrix = get_distance_matrix(ensemble1, ensemble2)
            with open(args.distance_matrix, 'wb') as f:
                np.save(f, distance_matrix)
    else:
        distance_matrix = get_distance_matrix(ensemble1, ensemble2)

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

    if args.dims == 1:
        ensemble1_data = transformed[:ensemble1_size].T[0]
        ensemble2_data = transformed[ensemble1_size:].T[0]

        kernel1 = gaussian_kde(ensemble1_data)
        kernel2 = gaussian_kde(ensemble2_data)

        xmin, xmax = np.min(transformed), np.max(transformed)
        size = xmax - xmin

        while kernel1.integrate_box(xmin, xmax) < 0.99 or kernel2.integrate_box(xmin, xmax) < 0.99:
            xmin -= 0.1 * size
            xmax += 0.1 * size

        grid = np.linspace(xmin, xmax, 500)

        if args.plot:
            plot_1d(kernel1, kernel2, ensemble1_data, ensemble2_data, grid)
    else:
        ensemble1_data = transformed[:ensemble1_size, :].T
        ensemble2_data = transformed[ensemble1_size:, :].T

        try:
            kernel1 = gaussian_kde(ensemble1_data)
            kernel2 = gaussian_kde(ensemble2_data)
        except np.linalg.LinAlgError:
            print('Cannot create a kernel, try lower dimension (-d 1)', file=sys.stderr)
            sys.exit(1)

        xmin, ymin = np.min(transformed, 0)
        xmax, ymax = np.max(transformed, 0)

        xsize = xmax - xmin
        ysize = ymax - ymin

        while kernel1.integrate_box([xmin, ymin], [xmax, ymax]) < 0.99 or kernel2.integrate_box([xmin, ymin], [xmax, ymax]) < 0.99:
            xmin -= 0.1 * xsize
            ymin -= 0.1 * ysize
            xmax += 0.1 * xsize
            ymax += 0.1 * ysize

        print('kernel1', kernel1.integrate_box([xmin, ymin], [xmax, ymax]))
        print('kernel2', kernel2.integrate_box([xmin, ymin], [xmax, ymax]))

        xx, yy = np.mgrid[xmin:xmax:500j, ymin:ymax:500j]
        grid = np.vstack([xx.ravel(), yy.ravel()])

        if args.plot:
            plot_2d(kernel1, kernel2, ensemble1_data, ensemble2_data, grid, xx, xmin, xmax, ymin, ymax)

    print('Final distance = {:f}'.format(jensen_shannon_distance(kernel1, kernel2, grid)))


if __name__ == '__main__':
    main()
