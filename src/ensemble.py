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


class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ResultRMSD:
    def __init__(self, all_files, selected_files, assigment_files_and_weights, run, method):
        self.selected_files = selected_files
        self.all_files = all_files
        self.assigment_files_and_weights = assigment_files_and_weights
        self.run = run
        self.data_and_weights = []
        self.stats = []
        self.method = method

    def print_result(self):
        print(Colors.OKGREEN, 'Run:', self.run, Colors.ENDC)
        print('Selected:', self.assigment_files_and_weights)
        print('Results:\n')
        for r, s in zip(self.data_and_weights, self.stats):
            print('RMSD: {:5.3f} chi2: {:5.3f} data: {}'.format(s[1], s[0], r))

    def get_best_result(self):
        return min(stat[1] for stat in self.stats)


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
    print('List of the all used files ({}):\n'.format(len(all_files)))
    for i in range(len(all_files)):
        if (i + 1) % 7 == 0:
            print(all_files[i])
        else:
            print(all_files[i], '\t', end='')
    print('\n')


def prepare_directory(all_files, selected_files, tmpdir, method):
    pathlib.Path(tmpdir + '/pdbs').mkdir(parents=True, exist_ok=True)
    pathlib.Path(tmpdir + '/dats').mkdir(parents=True, exist_ok=True)
    pathlib.Path(tmpdir + '/method').mkdir(parents=True, exist_ok=True)
    pathlib.Path(tmpdir + '/results').mkdir(parents=True, exist_ok=True)

    if method == ('ensemble' or 'foxs'):
        for file in all_files:
            command = f'foxs {file}'
            return_value = subprocess.call(command, shell=True)
            if return_value:
                print(f'ERROR: ensemble failed.', file=sys.stderr)
                sys.exit(1)
            shutil.copy(file + '.dat', '{}/dats/'.format(tmpdir))
        for i, f in enumerate(all_files, start=1):
            shutil.copy(f, '{}/pdbs/{:02d}.pdb'.format(tmpdir, i))

    else:
        for file in all_files:
            # make files.dat from pdbs
            command = f'foxs {file}'
            return_value = subprocess.call(command, shell=True)
            if return_value:
                print(f'ERROR: Foxs failed.', file=sys.stderr)
                sys.exit(1)
            shutil.copy(file, '{}/pdbs/'.format(tmpdir))
            shutil.copy(file + '.dat', '{}/dats/'.format(tmpdir))


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
            file.write('{:5.3f} {} 0\n'.format(q, y))

    adderror("../data/exp.dat", tmpdir + '/method/curve')


def ensemble_fit(all_files, tmpdir, repeat):
    # RUN ensemble
    ensemble_fit_binary = '/storage/brno3-cerit/home/krab1k/saxs-ensamble-fit/core/ensamble-fit'
    command = f'{ensemble_fit_binary} -L -p {tmpdir}/pdbs/ -n {len(all_files)} -m {tmpdir}/method/curve.modified.dat'
    shutil.copy('../test/result', tmpdir)
    print(Colors.OKBLUE + 'Command for ensemble fit \n' + Colors.ENDC, command, '\n')
    # return_value = subprocess.call(command, shell=True)
    # if return_value:
    #    print(f'ERROR: ensemble failed', file = sys.stderr)
    #    sys.exit(1)

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
    for chi2, weights in result_chi_and_weights_ensemble:
        for i, weight in enumerate(weights):
            if weight >= 0.001:
                ensemble_results.append((chi2, all_files[i], weight, repeat))

    return ensemble_results


def multifox(all_files, tmpdir, repeat):
    # RUN Multi_foxs
    files_for_multifox = ' '.join(str(tmpdir + '/pdbs/' + e) for e in all_files)
    print(files_for_multifox)
    command = 'multi_foxs {dir}/method/curve.modified.dat {pdb}'.format(dir=tmpdir, pdb=files_for_multifox)
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
            for line in file:
                # 1 |  3.05 | x1 3.05 (0.99, 0.20)
                #    0   | 1.000 (1.000, 1.000) | /tmp/tmpnz7dbids/pdbs/mod13.pdb  (0.062)
                if not line.startswith(' '):
                    if ' |' in line:
                        chi2 = float(line.split('|')[1])
                else:
                    weight = line[line.index('|') + 1:line.index('(')]
                    structure = line.split('pdbs/')[1].split('(')[0].strip()
                    result.append((chi2, structure, weight, repeat))
    if len(multifoxs_files) != 1:
        print(Colors.WARNING + '\nNot implemented now for more than one ensemble' + Colors.ENDC)
        sys.exit(0)
    # result = [(2.46, 'mod13.pdb', ' 1.000 '), (3.39, 'mod02.pdb', ' 1.000 '), (4.28, 'mod04.pdb', ' 1.000 ')]
    return result


def gajoe(all_files, tmpdir, repeat):
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
                file_gajoe.write(' {} {} {}\n'.format(x, b, c))
    # S values    num_lines
    # 0.000000E+00
    # ------
    #  Curve no.     1
    # 0.309279E+08

    num_lines = sum(1 for line in open(all_files[0] + ".dat")) - 2
    print(num_lines)
    with open(tmpdir + '/method/juneom.eom', 'w') as file1:
        file1.write('    S values   {} \n'.format(num_lines))
        with open(all_files[0] + ".dat") as file2:
            for line in file2:
                if line.startswith('#'):
                    continue
                data = float(line.split()[0])
                lineformat = ff.FortranRecordWriter('(1E14.6)')
                b = lineformat.write([data])
                file1.write('{}\n'.format(b))
        for i, filename in enumerate(all_files, start=1):
            with open(filename + ".dat") as file2:
                file1.write('Curve no.     {} \n'.format(i))
                for line in file2:
                    if line.startswith('#'):
                        continue
                    data1 = float(line.split()[1])
                    lineformat = ff.FortranRecordWriter('(1E14.6)')
                    b = lineformat.write([data1])
                    file1.write('{}\n'.format(b))
    command = 'yes | gajoe ./method/curve_gajoe.dat -i=./method/juneom.eom -t=5'.format()
    print(command)
    return_value = subprocess.check_call(command, cwd=tmpdir, shell=True)
    if return_value:
        print(f'ERROR: GAJOE failed', file=sys.stderr)
        sys.exit(1)
    # process results from gajoe (/GAOO1/curve_1/
    chi2 = None
    number = []
    weight = []
    order_of_curve = []
    with open(tmpdir + '/GA001/curve_1/logFile_001_1.log') as file_gajoe:
        for line in file_gajoe:
            if '-- Chi^2 : ' in line:
                chi2 = float(line.split(':')[1])
    # curve                   weight
    # 00002ethod/juneom.pd ~0.253.00
    # 00003ethod/juneom.pd ~0.172.00
            if ') 0' in line:
                    order_of_curve.append(int(line.split()[1][:5])-1) # number of curves in logfile > 1!!!
                    weight.append(float(line.split()[4][1:6]))

    #print(order_of_curve)
    order_weight = list(zip(order_of_curve, weight))
    result_chi_structure_weights_repeat = []
    #print(order_weight)
    #print(all_files)
    for i, file in enumerate(all_files):
        if i in order_of_curve:
            result_chi_structure_weights_repeat.append((chi2, all_files[i], weight[order_of_curve.index(i)], repeat))
            #print(result_chi_structure_weights_repeat)
    # [(1.255, 'mod06.pdb', 0.102, 1), (1.255, 'mod01.pdb', 0.153, 1), (1.255, 'mod20.pdb', 0.204, 1)]
    return (result_chi_structure_weights_repeat)


def process_result(tolerance, result_chi_structure_weights_repeat, k_options, selected_files, result):
    minimum = min(result_chi_structure_weights_repeat)
    maximum = minimum[0] * (1 + tolerance)
    final_result = []
    stats = []
    all_results = []

    # TODO biopython
    molecule_A = selected_files[0]
    for chi2, structure, weight, repeat in result_chi_structure_weights_repeat:
        if float(chi2) <= maximum:
            if float(weight) >= 0.001:
                final_result.append((weight, structure, repeat))
            sum_rmsd = 0
            if k_options == 1:
                structure_BIO = Bio.PDB.PDBParser().get_structure(structure, structure)
                ref_model = structure_BIO[0]
                print(ref_model)
                superimposer = Bio.PDB.Superimposer()
                superimposer.set_atoms(ref_model, structure)
                print(Colors.OKBLUE + ' \nRMSD pymol' + Colors.ENDC, sum_rmsd, '\n')
                stats.append((chi2, sum_rmsd))
                all_results.append(final_result)
            else:
                print('\n Not implemented now')
                sys.exit(1)
            print('\n \nResult from multifox for pymol \n')
    for weight, structure in final_result:
        print('Structure {} and weight {} \n'.format(weight, structure))
    print('len', len(all_results))
    result.stats = stats
    result.data_and_weights = all_results


def final_statistic(args, all_results):
    print(Colors.HEADER + '\nFINAL STATISTICS \n' + Colors.ENDC)
    rmsd = [result.get_best_result() for result in all_results]
    print('RMSD = ', np.mean(rmsd), '±', np.std(rmsd))
    print('Number of runs', args.repeat, '\n')


def main():
    random.seed(1)
    np.random.seed(1)
    all_results = []
    args = get_argument()
    os.chdir(args.mydirvariable)
    list_pdb_file = find_pdb_file(args.mydirvariable)
    test_argument(args.n_files, args.k_options, list_pdb_file, args.tolerance)
    result_chi_structure_weights_repeat = []
    for i in range(args.repeat):
        tmpdir = tempfile.mkdtemp()
        print(Colors.OKGREEN + 'RUN {}/{}'.format(i + 1, args.repeat) + Colors.ENDC, '\n')
        all_files = random.sample(list_pdb_file, args.n_files)
        # copy to pds
        selected_files = random.sample(all_files, args.k_options)
        # copy to dats
        weights = np.random.dirichlet(np.ones(args.k_options), size=1)[0]
        files_and_weights = list(zip(selected_files, weights))
        # copy to metods
        prepare_directory(all_files, selected_files, tmpdir, args.method)
        make_curve_for_experiment(files_and_weights, tmpdir)

        print(Colors.OKBLUE + '\nCreated temporary directory \n' + Colors.ENDC, tmpdir, '\n')
        print(Colors.OKBLUE + 'Method' + Colors.ENDC, '\n')
        print(args.method, '\n')
        if args.verbose:
            print_parameters_verbose(args, list_pdb_file, all_files)
        result = ResultRMSD(all_files, selected_files, files_and_weights, i + 1, args.method)
        if args.method == 'ensemble':
            result_chi_structure_weights_repeat = ensemble_fit(all_files, tmpdir, args.repeat)

        if args.method == 'eom':
            result_chi_structure_weights_repeat = gajoe(all_files, tmpdir, args.repeat)

        if args.method == 'foxs':
            result_chi_structure_weights_repeat = multifox(all_files, tmpdir, args.repeat)

        process_result(args.tolerance, result_chi_structure_weights_repeat,
                       args.k_options, selected_files, result)
    print('hromadny vysledek', result_chi_structure_weights_repeat)

    if not args.preserve:
        shutil.rmtree(tmpdir)

    for result in all_results:
        result.print_result()
    final_statistic(args, all_results)


if __name__ == '__main__':
    main()
