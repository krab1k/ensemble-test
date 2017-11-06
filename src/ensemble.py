#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from os import listdir
import numpy as np
import fortranformat as ff
from adderror import adderror
import pathlib
import Bio.PDB

ENSEMBLE_BINARY = '/home/saxs/saxs-ensamble-fit/core/ensemble-fit'


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Run:
    def __init__(self, all_files, selected_files, weights, run, method):
        # experiment
        self.all_files = all_files
        self.selected_files = selected_files
        self.weights = weights
        self.method = method
        self.run = run
        # results from experiment
        self.results = []

    def print_result(self, args):
        print(Colors.OKGREEN, 'Run:', self.run, Colors.ENDC)
        print(Colors.OKBLUE + '\nSelected structure:\n' + Colors.ENDC)
        for structure, weight in list(zip(self.selected_files, self.weights)):
            print(f'structure: {structure} weight: {weight:.3f} \n')
        if args.verbose:
            print(Colors.OKBLUE + '\nResults:\n ' + Colors.ENDC)
            for sumrmsd, chi2, data in self.results:
                print(f'RMSD: {sumrmsd:.3f} Chi2: {chi2:.3f}\n')
                for structure, weight in data:
                    print(f'structure: {structure} weight: {weight:.3f} \n')

    def get_best_result(self):
        return min(rmsd for rmsd, _, _ in self.results)


def get_argument():
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", dest="mydirvariable",
                        help="Choose dir", metavar="DIR", required=True)

    parser.add_argument("-n", metavar='N', type=int,
                        dest="n_files",
                        help="Number of selected structure",
                        required=True)

    parser.add_argument("-k", metavar='K', type=int,
                        dest="k_options",
                        help="Number of possibility structure, less then selected files",
                        required=True)

    parser.add_argument("-r", metavar='R', type=int,
                        dest="repeat", help="Number of repetitions",
                        default=1)
    parser.add_argument("--verbose", help="increase output verbosity",
                        action="store_true")

    parser.add_argument("--tolerance", type=float, dest="tolerance",
                        help="pessimist (0) or optimist (0 < x <1) result",
                        default=0)

    parser.add_argument("--preserve", help="preserve temporary directory",
                        action="store_true")

    parser.add_argument("--method", help="choose method ensamble, eom, foxs",
                        choices=['ensemble', 'eom', 'foxs'])

    return parser.parse_args()


def find_pdb_file(mydirvariable):
    pdb_files = []
    files = listdir(mydirvariable)
    for line in files:
        line = line.rstrip()
        if re.search('.pdb$', line):
            pdb_files.append(line)

    return pdb_files


def test_argument(n_files, k_options, list_pdb_file, tolerance):
    if len(list_pdb_file) < n_files:
        print(Colors.WARNING + "Number of pdb files is ONLY" + Colors.ENDC, len(list_pdb_file), '\n')
        sys.exit(1)
    if k_options > n_files:
        print(Colors.WARNING + "Number of selected structure is ONLY" + Colors.ENDC, n_files, '\n')
        sys.exit(1)
    if tolerance > 1:
        print('Tolerance should be less then 1.')
        sys.exit(1)


def print_parameters_verbose(args, list_pdb_file, all_files):
    print(Colors.OKBLUE + 'Parameters \n' + Colors.ENDC)
    print('Working directory', os.getcwd(), '\n')
    print('Tolerance', args.tolerance, '\n')
    print('Total number of available pdb files in the directory', len(list_pdb_file), '\n')
    print(f'List of the all used files ({len(all_files)}):\n')
    for i in range(len(all_files)):
        if (i + 1) % 7 == 0:
            print(all_files[i])
        else:
            print(all_files[i], '\t', end='')
    print('\n')


def prepare_directory(all_files, selected_files, tmpdir, method):
    pathlib.Path(tmpdir + '/pdbs').mkdir(parents=True, exist_ok=True)
    if method == 'ensemble':
        pathlib.Path(tmpdir + '/pdbs/ensembles').mkdir(parents=True, exist_ok=True)

    pathlib.Path(tmpdir + '/dats').mkdir(parents=True, exist_ok=True)
    pathlib.Path(tmpdir + '/method').mkdir(parents=True, exist_ok=True)
    pathlib.Path(tmpdir + '/results').mkdir(parents=True, exist_ok=True)
    # prepare 'file'.dat and copy to /dats/
    for file in all_files:
        command = f'foxs {file}'
        return_value = subprocess.call(command, shell=True)
        if return_value:
            print(f'ERROR: Foxs failed.', file=sys.stderr)
            sys.exit(1)
        shutil.copy(file + '.dat', f'{tmpdir}/dats/')

    if method == 'ensemble':  # format 01.pdb, 02.pdb as input for ensemble
        for i, f in enumerate(all_files, start=1):
            shutil.copy(f, f'{tmpdir}/pdbs/ensembles/{i:02d}.pdb')

    for file in all_files:  # not strict format for pdbs file
        shutil.copy(file, f'{tmpdir}/pdbs/')


def make_curve_for_experiment(files_and_weights, tmpdir):
    files = [filename for filename, weight in files_and_weights]
    print(Colors.OKBLUE + 'Data for experiment \n' + Colors.ENDC)
    for data, weight in files_and_weights:
        print('structure', data, 'weight', round(weight, 3), '\n')
    qs = np.linspace(0, 0.5, 501)
    curves = {}
    for filename in files:
        with open(filename + '.dat') as file:
            data = []
            for line in file:
                if line.startswith('#'):
                    continue
                data.append(line.split()[1])
            curves[filename] = np.array(data, dtype=float)

    result_curve = np.zeros(501, dtype=float)

    for filename, weight in files_and_weights:
        result_curve += curves[filename] * weight

    with open(tmpdir + '/method/curve', 'w') as file:
        for q, y in zip(qs, result_curve):
            file.write(f'{q:5.3f} {y} 0\n')

    adderror("../data/exp.dat", tmpdir + '/method/curve')


def ensemble_fit(all_files, tmpdir):
    # RUN ensemble
    currdir = os.getcwd()
    os.chdir(tmpdir)
    command = f'{ENSEMBLE_BINARY} -L -p {tmpdir}/pdbs/ensembles/ -n {len(all_files)} -m {tmpdir}/method/curve.modified.dat'
    print(Colors.OKBLUE + 'Command for ensemble fit \n' + Colors.ENDC, command, '\n')
    return_value = subprocess.call(command, shell=True)
    if return_value:
        print(f'ERROR: ensemble failed', file = sys.stderr)
        sys.exit(1)

    os.chdir(currdir)
    # Process with result from ensemble
    result_chi_and_weights_ensemble = []
    # 5000
    # 1.08e+01,0.952,2.558,4.352610,0.000,0.300,0.000,0.000,0.000,0.000,0.000,0.092,0.000,0.908
    # 1.08e+01,0.950,2.558,4.752610,0.000,0.100,0.000,0.000,0.000,0.000,0.000,0.092,0.000,0.908
    with open(tmpdir + '/result', 'r') as f:
        next(f)

        for line in f:
            line = line.rstrip()
            value_of_chi2 = float(line.split(',')[3])
            values_of_index_result = [float(value) for value in line.split(',')[4:]]
            result_chi_and_weights_ensemble.append((value_of_chi2, values_of_index_result))
    ensemble_results = []
    structure = []
    for chi2, weights in result_chi_and_weights_ensemble:
        for i, weight in enumerate(weights):
            if weight >= 0.001:
                structure.append((all_files[i], weight))
        ensemble_results.append((chi2, structure))
    return ensemble_results

    # ((chi2, [('mod10.pdb', 0.3), ('mod15.pdb', 0.7)]),(chi2(strucutre, weight),(strucutre, weight)))


def multifox(all_files, tmpdir):
    # RUN Multi_foxs
    files_for_multifox = ' '.join(str(tmpdir + '/pdbs/' + e) for e in all_files)
    print(files_for_multifox)
    command = f'multi_foxs {tmpdir}/method/curve.modified.dat {files_for_multifox}'
    return_value = subprocess.check_call(command, cwd=tmpdir, shell=True)
    if return_value:
        print(f'ERROR: multifox failed', file=sys.stderr)
        sys.exit(1)
    # Process with result from Multi_foxs
    multifoxs_files = []
    files = listdir(tmpdir)
    for line in files:
        line = line.rstrip()
        if re.search('\d.txt$', line):
            multifoxs_files.append(line)
    result = []
    chi2 = 0
    for filename in multifoxs_files:
        with open(tmpdir + '/' + filename) as file:
            weight_structure = []
            for line in file:
                # 1 |  3.05 | x1 3.05 (0.99, 0.20)
                #    0   | 1.000 (1.000, 1.000) | /tmp/tmpnz7dbids/pdbs/mod13.pdb  (0.062)
                if not line.startswith(' '):
                    if weight_structure:
                        result.append((chi2, weight_structure))
                    chi2 = float(line.split('|')[1])
                    weight_structure = []

                else:
                    weight = float(line[line.index('|') + 1:line.index('(')])
                    structure = line.split('pdbs/')[1].split('(')[0].strip()
                    weight_structure.append((structure, weight))
            result.append((chi2, weight_structure))
    # ((chi2, [('mod10.pdb', 0.3), ('mod15.pdb', 0.7)]),(chi2, [(strucutre, weight),(strucutre, weight),...)])
    return result


def gajoe(all_files, tmpdir):
    # Angular axis m01000.sax             Datafile m21000.sub         21-Jun-2001
    # .0162755E+00 0.644075E+03 0.293106E+02
    with open(tmpdir + '/method/curve_gajoe.dat', 'w') as file_gajoe:
        file_gajoe.write(' Angular axis m01000.sax             Datafile m21000.sub         21-Jun-2001\n')
        lineformat = ff.FortranRecordWriter('(1E12.6)')
        with open(tmpdir + '/method/curve.modified.dat') as file1:
            for line in file1:
                if not line.strip():
                    break
                data1 = float(line.split()[0])
                data2 = float(line.split()[1])
                data3 = float(line.split()[2])
                a = lineformat.write([data1])
                x = a[1:]
                b = lineformat.write([data2])
                c = lineformat.write([data3])
                file_gajoe.write(f' {x} {b} {c}\n')
    # S values    num_lines
    # 0.000000E+00
    # ------
    #  Curve no.     1
    # 0.309279E+08
    num_lines = sum(1 for line in open(all_files[0] + ".dat")) - 2
    print(num_lines)
    with open(tmpdir + '/method/juneom.eom', 'w') as file1:
        file1.write(f'    S values   {num_lines} \n')
        with open(all_files[0] + ".dat") as file2:
            for line in file2:
                if line.startswith('#'):
                    continue
                data = float(line.split()[0])
                lineformat = ff.FortranRecordWriter('(1E14.6)')
                b = lineformat.write([data])
                file1.write(f'{b}\n')
        for i, filename in enumerate(all_files, start=1):
            with open(filename + ".dat") as file2:
                file1.write(f'Curve no.     {i} \n')
                for line in file2:
                    if line.startswith('#'):
                        continue
                    data1 = float(line.split()[1])
                    lineformat = ff.FortranRecordWriter('(1E14.6)')
                    b = lineformat.write([data1])
                    file1.write(f'{b}\n')
    command = f'yes | gajoe ./method/curve_gajoe.dat -i=./method/juneom.eom -t=5'
    print(command)
    return_value = subprocess.check_call(command, cwd=tmpdir, shell=True)
    if return_value:
        print(f'ERROR: GAJOE failed', file=sys.stderr)
        sys.exit(1)
    # process results from gajoe (/GAOO1/curve_1/
    chi2 = None
    structure_weight = []
    m = re.compile('^\s*\d+\)')
    with open(tmpdir + '/GA001/curve_1/logFile_001_1.log') as file_gajoe:
        for line in file_gajoe:
            if '-- Chi^2 : ' in line:
                chi2 = float(line.split(':')[1])
                # curve                   weight
                # 00002ethod/juneom.pd ~0.253.00
                # 00003ethod/juneom.pd ~0.172.00
            p = m.search(line)
            if p:
                index = int(line.split()[1][:5]) - 1
                weight = float(line.split()[4][1:6])
                structure_weight.append((all_files[index], weight))

    return [(chi2, structure_weight)]
    # ([chi2,[(structure, weight), (structure,weight), (structure, weight),... ], [chi2,(),...])


def process_result(tolerance, result_chi_structure_weights, selected_files, run, tmpdir):
    minimum = min(chi2 for chi2, _ in result_chi_structure_weights)
    maximum = minimum * (1 + tolerance)

    all_results = []
    for chi2, names_and_weights in result_chi_structure_weights:
        sum_rmsd = 0
        if float(chi2) <= maximum:
            assert len(selected_files) == 1
            reference_structure = Bio.PDB.PDBParser(QUIET=True).get_structure('reference',
                                                                              tmpdir + '/pdbs/' + selected_files[0])
            for structure, weight in names_and_weights:
                structure_1 = Bio.PDB.PDBParser(QUIET=True).get_structure('alternative', tmpdir + '/pdbs/' + structure)
                superimposer = Bio.PDB.Superimposer()
                superimposer.set_atoms(list(reference_structure.get_atoms()), list(structure_1.get_atoms()))
                sum_rmsd += superimposer.rms * weight
            # print(Colors.OKBLUE + ' \nweighted RMSD = ' + Colors.ENDC, sum_rmsd, '\n')

            all_results.append((sum_rmsd, chi2, names_and_weights))  # add results only with RMSD within selected limit

    run.results = all_results
    return run


def final_statistic(runs):
    print(Colors.HEADER + '\nFINAL STATISTICS \n' + Colors.ENDC)
    rmsd = [result.get_best_result() for result in runs]
    print('Number of runs: ', len(runs))
    print('RMSD = {:.3f} ± {:.3f}'.format(np.mean(rmsd), np.std(rmsd)))


def main():
    random.seed(1)
    np.random.seed(1)
    args = get_argument()
    os.chdir(args.mydirvariable)
    list_pdb_file = find_pdb_file(args.mydirvariable)
    test_argument(args.n_files, args.k_options, list_pdb_file, args.tolerance)
    result_chi_structure_weights = []
    all_runs = []
    for i in range(args.repeat):
        tmpdir = tempfile.mkdtemp()
        print(Colors.OKGREEN + f'RUN {i+1}/{args.repeat}' + Colors.ENDC, '\n')
        all_files = random.sample(list_pdb_file, args.n_files)
        # copy to pds
        selected_files = random.sample(all_files, args.k_options)
        # copy to dats
        weights = np.random.dirichlet(np.ones(args.k_options), size=1)[0]
        files_and_weights = list(zip(selected_files, weights))
        # copy to methods
        prepare_directory(all_files, selected_files, tmpdir, args.method)
        make_curve_for_experiment(files_and_weights, tmpdir)
        print(Colors.OKBLUE + '\nCreated temporary directory \n' + Colors.ENDC, tmpdir, '\n')
        print(Colors.OKBLUE + 'Method' + Colors.ENDC, '\n')
        print(args.method, '\n')
        if args.verbose:
            print_parameters_verbose(args, list_pdb_file, all_files)
        run = Run(all_files, selected_files, weights, i + 1, args.method)
        if args.method == 'ensemble':
            result_chi_structure_weights = ensemble_fit(all_files, tmpdir)

        elif args.method == 'eom':
            result_chi_structure_weights = gajoe(all_files, tmpdir)

        elif args.method == 'foxs':
            result_chi_structure_weights = multifox(all_files, tmpdir)

        run = process_result(args.tolerance, result_chi_structure_weights, selected_files, run, tmpdir)

        all_runs.append(run)

    if not args.preserve:
        shutil.rmtree(tmpdir)

    for run in all_runs:
        run.print_result(args)

    final_statistic(all_runs)


if __name__ == '__main__':
    main()
