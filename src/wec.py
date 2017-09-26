import Bio.PDB
import numpy as np
import argparse
import itertools
import sys
from sklearn import manifold
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def align_structures(ensemble1, ensemble2):
    reference = ensemble1[0]
    for model in itertools.chain(list(ensemble1)[1:], list(ensemble2)):
        superimposer = Bio.PDB.Superimposer()
        superimposer.set_atoms(list(reference.get_atoms()), list(model.get_atoms()))
        superimposer.apply(list(model.get_atoms()))


def get_data(ensemble1, ensemble2):
    n = 3 * len(list(ensemble1[0].get_atoms()))
    m = len(list(ensemble1)) + len(list(ensemble2))

    data = np.empty((m, n), dtype=np.float_)
    for row, model in enumerate(itertools.chain(ensemble1, ensemble2)):
        for i, atom in enumerate(model.get_atoms()):
            data[row, 3 * i:3 * i + 3] = atom.get_coord()
    return data


def kullback_leibler_divergence(p_x, p_y):
    return np.sum(p_x * np.log(p_x / p_y))


def jensen_shannon_divergence(kernel1, kernel2, grid):
    p_x = kernel1(grid)
    p_y = kernel2(grid)
    m = 0.5 * (p_x + p_y)
    return 0.5 * (kullback_leibler_divergence(p_x, m) + kullback_leibler_divergence(p_y, m))


def load_weights(filename):
    try:
        weights = []
        with open(filename) as f:
            weights.extend(float(v) for v in f.readline().split())
            weights.extend(float(v) for v in f.readline().split())
            return weights
    except IOError:
        print(f'Cannot open file with weights {filename}', file=sys.stderr)
        sys.exit(1)


def adjust_data_by_weights(data, weights):
    tmp = []
    for d, w in zip(data.T, weights):
        print(d.shape, d)
        tile = np.tile(d.reshape((d.shape[0], 1)), int(w * 10))
        tmp.append(tile)

    return np.hstack(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ensemble1')
    parser.add_argument('ensemble2')
    parser.add_argument('-d', '--dims', default=2, choices=[1, 2], type=int)
    parser.add_argument('-w', '--weights', type=str)

    args = parser.parse_args()

    if args.weights:
        weights = load_weights(args.weights)


    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    ensemble1 = pdb_parser.get_structure('en1', args.ensemble1)
    ensemble2 = pdb_parser.get_structure('en2', args.ensemble2)

    ensemble1_size = len(ensemble1.get_list())
    ensemble2_size = len(ensemble2.get_list())

    align_structures(ensemble1, ensemble2)
    data = get_data(ensemble1, ensemble2)

    isomap = manifold.Isomap(n_neighbors=3, n_components=args.dims)
    np.random.seed(0)
    transformed = isomap.fit_transform(data)

    #if args.weights:
    weights = list(v / 100 for v in range(ensemble1_size + ensemble2_size))
    adjust_data_by_weights(transformed, weights)

    if args.dims == 1:
        ensemble1_data = transformed[:ensemble1_size].T[0]
        ensemble2_data = transformed[ensemble1_size:].T[0]
        kernel1 = gaussian_kde(ensemble1_data)
        kernel2 = gaussian_kde(ensemble2_data)

        xmin = np.concatenate((ensemble1_data, ensemble2_data)).min()
        xmax = np.concatenate((ensemble1_data, ensemble2_data)).max()
        grid = np.linspace(xmin, xmax, 100)

        plt.plot(grid, kernel1.evaluate(grid), 'r')
        plt.plot(grid, kernel2.evaluate(grid), 'b')

        plt.scatter(ensemble1_data, np.full(ensemble1_size, -0.01), c='r', marker='.', alpha=0.5)
        plt.scatter(ensemble2_data, np.full(ensemble2_size, -0.02), c='b', marker='.', alpha=0.5)
        plt.show()

    else:
        ensemble1_data = transformed[:ensemble1_size, :].T
        ensemble2_data = transformed[ensemble1_size:, :].T
        kernel1 = gaussian_kde(ensemble1_data)
        kernel2 = gaussian_kde(ensemble2_data)

        xmin, ymin = np.min(transformed, 0)
        xmax, ymax = np.max(transformed, 0)
        xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        grid = np.vstack([xx.ravel(), yy.ravel()])

        res1 = kernel1.evaluate(grid)
        res2 = kernel2.evaluate(grid)

        z1 = np.reshape(res1.T, xx.shape)
        z2 = np.reshape(res2.T, xx.shape)

        fig, ax = plt.subplots()
        ax.imshow(np.rot90(z1), cmap=plt.cm.Reds, extent=[xmin, xmax, ymin, ymax], alpha=0.5)
        ax.imshow(np.rot90(z2), cmap=plt.cm.Blues, extent=[xmin, xmax, ymin, ymax], alpha=0.5)

        plt.scatter(*ensemble1_data, c='r', alpha=0.5)
        plt.scatter(*ensemble2_data, c='b', alpha=0.5)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        plt.show()

    print(jensen_shannon_divergence(kernel1, kernel2, grid))


if __name__ == '__main__':
    main()
