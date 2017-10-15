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
    def __init__(self, all_files, selected_files, assigment_files_and_weights, run):
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

def make_curve_for_experiment(files_and_weights, tmpdirname):
    files = [filename for filename, weight in files_and_weights]
    print(Colors.OKBLUE + 'Data for ensamble fit \n' + Colors.ENDC)
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

    with open(tmpdirname + '/curve', 'w') as file:
        for q, y in zip(qs, result_curve):
            file.write('{:5.3f} {} 0\n'.format(q, y))

    adderror("../data/exp.dat", tmpdirname + '/curve')


def ensemble_fit(all_files, tmpdirname):
    print(Colors.OKBLUE + '\nCreated temporary directory \n' + Colors.ENDC, tmpdirname, '\n')
    for i, f in enumerate(all_files, start=1):
        shutil.copy(f, '{}/{:02d}.pdb.dat'.format(tmpdirname, i))
    ensemble_fit_binary = '/storage/brno3-cerit/home/krab1k/saxs-ensamble-fit/core/ensamble-fit'
    command = f'{ensemble_fit_binary} -L -p {tmpdir}/ -n {len(selected_files)} -m {tmpdir}/curve.modified.dat'
    shutil.copy('../test/result', tmpdirname)
    print(Colors.OKBLUE + 'Command for ensemble fit \n' + Colors.ENDC, command, '\n')
    # subprocess.call(command, shell=True)
    return work_with_result_from_ensemble(tmpdirname)

def multifox(all_files, tmpdirname):
    print(Colors.OKBLUE + '\nCreated temporary directory \n' + Colors.ENDC, tmpdirname, '\n')
    for file in all_files:
        shutil.copy(file, '{}'.format(tmpdirname))
    files_for_multifox = ' '.join(str(e) for e in all_files)
    command = 'multi_foxs curve.modified.dat {pdb}'.format(dir=tmpdirname, pdb=files_for_multifox)
    subprocess.check_call(command, cwd=tmpdirname, shell=True)


def gajoe(all_files, tmpdirname):
    with open(tmpdirname + '/curve_gajoe.dat', 'w') as file_gajoe:
        file_gajoe.write(' Angular axis m01000.sax             Datafile m21000.sub         21-Jun-2001\n')
        lineformat = ff.FortranRecordWriter('(1E12.6)')
        with open(tmpdirname + '/curve.modified.dat') as file1:
            for line in file1:
                data1 = float(line.split()[0])
                data2 = float(line.split()[1])
                data3 = float(line.split()[2])
                a = lineformat.write([data1])
                x = a[1:]
                b = lineformat.write([data2])
                c = lineformat.write([data3])
                file_gajoe.write(' {} {} {}\n'.format(x, b, c))
    print(Colors.OKBLUE + '\nCreated temporary directory \n' + Colors.ENDC, tmpdirname, '\n')
    with open(tmpdirname + '/juneom.eom', 'w') as file1:
        file1.write('    S values    51 \n')
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
    command = 'gajoe curve.modified.dat -i=juneom.eom -t=5'.format(dir=tmpdirname)
    print(command)
    # subprocess.check_call(command, cwd=tmpdirname, shell=True)
    return ()


def work_with_result_from_ensemble(tmpdirname):
    result_chi_and_weights_ensemble = []
    with open(tmpdirname + '/result', 'r') as f:
        next(f)

        for line in f:
            line = line.rstrip()
            value_of_chi2 = float(line.split(',')[3])
            values_of_index_result = [float(value) for value in line.split(',')[4:]]
            result_chi_and_weights_ensemble.append((value_of_chi2, values_of_index_result))

    return result_chi_and_weights_ensemble


def work_with_result_from_multifox(tmpdirname):
    multifoxs_files = []
    files = listdir(tmpdirname)
    for line in files:
        line = line.rstrip()
        if re.search('\d.txt$', line):
            multifoxs_files.append(line)
    result = []
    chi2 = 0
    for filename in multifoxs_files:
        with open(filename) as file:
            for line in file:
                # 1 |  3.05 | x1 3.05 (0.99, 0.20)
                #    0   | 1.000 (1.000, 1.000) | mod13.pdb (0.062)
                if not line.startswith(' '):
                    if ' |' in line:
                        chi2 = float(line.split('|')[1])
                else:
                    weight = line[line.index('|') + 1:line.index('(')]
                    structure = line.split('|')[2].split('(')[0].strip()
                    result.append((chi2, structure, weight))
    if len(multifoxs_files) != 1:
        print(Colors.WARNING + '\nNot implemented now for more than one ensemble' + Colors.ENDC)
        sys.exit(0)
    return result


def process_result_multifox(tolerance, tmpdirname,
                            result_chi_and_weights_multifox, k_options,
                            selected_files, result):
    minimum = min(result_chi_and_weights_multifox)  # map(min, zip(*result_chi_and_weights_multifox))
    maximum = minimum[0] * (1 + tolerance)
    multifox_results = []
    stats = []
    all_results = []
    for chi2, structure, weight in result_chi_and_weights_multifox:
        if float(chi2) <= maximum:
            if float(weight) >= 0.001:
                multifox_results.append((weight, structure))

            sum_rmsd = 0
            if k_options == 1:
                for weight1, structure1 in multifox_results:
                    sum_rmsd = + rmsd_pymol(selected_files[0], structure1, tmpdirname) * float(weight1)

                print(Colors.OKBLUE + ' \nRMSD pymol' + Colors.ENDC, sum_rmsd, '\n')
                stats.append((chi2, sum_rmsd))
                all_results.append(multifox_results)
            else:
                print('\n Not implemented now')
                sys.exit(1)
            print('\n \nResult from multifox for pymol \n')
    for weight, structure in multifox_results:
        print('Structure {} and weight {} \n'.format(weight, structure))
    print('len', len(all_results))
    result.stats = stats
    result.data_and_weights = all_results


def process_result_ensemble(tolerance, all_files, selected_files, result_chi_and_weights_ensemble, tmpdirname, result):
    minimum = min(result_chi_and_weights_ensemble)[0]
    maximum = minimum * (1 + tolerance)
    stats = []
    all_results = []
    for chi2, weights in result_chi_and_weights_ensemble:
        ensemble_results = []
        if chi2 <= maximum:
            for i, weight in enumerate(weights):
                if weight >= 0.001:
                    ensemble_results.append((weight, all_files[i]))

            sum_rmsd = 0
            for r in ensemble_results:
                sum_rmsd += rmsd_pymol(selected_files[0], r[1], tmpdirname) * float(r[0])

            print(Colors.OKBLUE + 'RMSD pymol' + Colors.ENDC, sum_rmsd, '\n')
            stats.append((chi2, sum_rmsd))
            all_results.append(ensemble_results)

    result.stats = stats
    result.data_and_weights = all_results


def rmsd_pymol(structure_1, structure_2, tmpdirname):
    shutil.copy(structure_1, tmpdirname)
    shutil.copy(structure_2, tmpdirname)
    if structure_1 == structure_2:
        rmsd = 0
    else:
        with open(tmpdirname + '/file_for_pymol.pml', 'w') as file_for_pymol:
            file_for_pymol.write("""
            load  {s1}
            load  {s2}
            align {s3}, {s4}
            quit
            """.format(s1=structure_1, s2=structure_2,
                       s3=os.path.splitext(structure_1)[0],
                       s4=os.path.splitext(structure_2)[0]))
        # out_pymol = subprocess.check_output("module add pymol-1.8.2.1-gcc; pymol -c file_for_pymol.pml | grep Executive:; module rm pymol-1.8.2.1-gcc", shell=True)
        command = ' pymol -c {dir}/file_for_pymol.pml | grep Executive:'.format(dir=tmpdirname)
        out_pymol = subprocess.check_output(command, shell=True)
        rmsd = float(out_pymol[out_pymol.index(b'=') + 1:out_pymol.index(b'(') - 1])
    print('RMSD ', structure_1, ' and ', structure_2, ' = ', rmsd)
    return rmsd


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

    for i in range(args.repeat):
        tmpdirname = tempfile.mkdtemp()
        print(Colors.OKGREEN + 'RUN {}/{}'.format(i + 1, args.repeat) + Colors.ENDC, '\n')
        all_files = random.sample(list_pdb_file, args.n_files)
        selected_files = random.sample(all_files, args.k_options)
        weights = np.random.dirichlet(np.ones(args.k_options), size=1)[0]
        files_and_weights = list(zip(selected_files, weights))
        make_curve_for_experiment(files_and_weights, tmpdirname)
        print(Colors.OKBLUE + 'Method' + Colors.ENDC, '\n')
        print(args.method, '\n')
        if args.verbose:
            print_parameters_verbose(args, list_pdb_file, all_files)
        if args.method == 'ensemble':
            result = RMSD_result(all_files, selected_files, files_and_weights, i + 1, args.method)
            result_chi_and_weights_ensemble = ensemble_fit(all_files, tmpdirname)
            if args.k_options == 1:
                process_result_ensemble(args.tolerance, all_files, selected_files, result_chi_and_weights_ensemble,
                                        tmpdirname, result)
            else:
                print(Colors.WARNING + 'k > 1 not implemented.\n' + Colors.ENDC)
            if not args.preserve:
                shutil.rmtree(tmpdirname)
            all_results.append(result)
        if args.method == 'eom':
            # if args.k_options == 1:
            gajoe(selected_files, tmpdirname)
            # else:    print('Not implemented now.')
            # sys.exit(1)
        if args.method == 'foxs':
            result =  ResultRMSD(all_files, selected_files, files_and_weights, i + 1, args.method)
            multifox(all_files, tmpdirname)
            result_chi_and_weights_multifox = work_with_result_from_multifox(tmpdirname)
            process_result_multifox(args.tolerance, tmpdirname, result_chi_and_weights_multifox,
                                    args.k_options, selected_files, result)
            if not args.preserve:
                shutil.rmtree(tmpdirname)
            all_results.append(result)

            # print('Not implemented now.')
            # sys.exit(1)
    for result in all_results:
        result.print_result()
    final_statistic(args, all_results)


if __name__ == '__main__':
    main()
