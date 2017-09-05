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

class RMSD_result:
    def __init__(self, repeat = 0, selected_files = '', all_files = '', files_from_ensamble = '',
                structure_and_rmsd = '', sum_rmsd = -100, files_and_weights = ''):
                self.repeat = repeat
                self.selected_files = selected_files
                self.all_files = all_files
                self.files_from_ensamble = files_from_ensamble
                self.structure_and_rmsd = structure_and_rmsd
                self.sum_rmsd = sum_rmsd
                self.files_and_weights = files_and_weights

    def print_result(self):
        print('vysledek', self.repeat)






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


def print_parameters_verbose(args, list_pdb_file, all_files, selected_files, files_and_weights):
    print(Colors.OKBLUE + 'Parameters \n' + Colors.ENDC)
    print('Working directory', os.getcwd(), '\n')
    print('Total number of available pdb files in the directory', len(list_pdb_file), '\n')
    print('Number of the all used files', args.n_files, '\n')
    print('List of the all used files \n')
    for i in range(len(all_files)):
        if (i + 1) % 7 == 0:
            print(all_files[i])
        else:
            print(all_files[i], '\t', end='')
    print('\n')
    print('Number of selected files', args.k_options, '\n')
    print('List of selected files \n')
    for i in range(len(selected_files)):
        if (i + 1) % 7 == 0:
            print(selected_files[i])
        else:
            print(selected_files[i], '\t', end='')
    print('\n')


def ensamble_fit(selected_files_for_ensamble, files_and_weights, tmpdirname):
    print(Colors.OKBLUE + '\nCreated temporary directory \n' + Colors.ENDC, tmpdirname, '\n')
    for i, f in enumerate(selected_files_for_ensamble, start=1):
        shutil.copy(f, '{}/{:02d}.pdb.dat'.format(tmpdirname, i))
    make_curve_for_experiment(files_and_weights, tmpdirname)
    command = '/storage/brno3-cerit/home/krab1k/saxs-ensamble-fit/core/ensamble-fit -L -p {dir}/ -n {n} -m {dir}/curve.modified.dat'.format(
        dir=tmpdirname, n=len(selected_files_for_ensamble))
    shutil.copy('../test/result', tmpdirname)
    print(Colors.OKBLUE + 'Command for ensamble fit \n' + Colors.ENDC, command, '\n')
    # subprocess.call(command, shell=True)

    return work_with_result_from_ensamble(tmpdirname)


def work_with_result_from_ensamble(tmpdirname):
    result_chi_and_weights = []
    with open(tmpdirname + '/result', 'r') as f:
        next(f)

        for line in f:
            line = line.rstrip()
            value_of_chi2 = float(line.split(',')[3])
            values_of_index_result = [float(value) for value in line.split(',')[4:]]
            result_chi_and_weights.append((value_of_chi2, values_of_index_result))

    return result_chi_and_weights


def do_result(tolerance, all_files, selected_files, result_chi_and_weights, tmpdirname):
    results = []
    minimum = min(result_chi_and_weights)[0]
    maximum = minimum * (1 + tolerance)
    for chi2, weights in result_chi_and_weights:
        if chi2 <= maximum:
            for i, weight in enumerate(weights):
                if weight >= 0.001:
                    results.append((chi2, weight, all_files[i]))

            sum_rmsd = 0
            for k in results:
                sum_rmsd += rmsd_pymol(selected_files[0], k[2], tmpdirname) * float(k[1])

            print(Colors.OKBLUE + 'RMSD pymol' + Colors.ENDC, sum_rmsd, '\n')


def make_curve_for_experiment(files_and_weights, tmpdirname):
    files = [filename for filename, weight in files_and_weights]
    print(Colors.OKBLUE+'Data for ensamble fit \n'+ Colors.ENDC)
    for data, weight in files_and_weights:
        print('structure',data, 'weight', round(weight,3), '\n')
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

def rmsd_pymol(structure_1, structure_2, tmpdirname):
    shutil.copy(structure_1, tmpdirname)
    shutil.copy(structure_2, tmpdirname)
    if structure_1 == structure_2:
        rmsd = 0
    else:
        with open(tmpdirname +'/file_for_pymol.pml', 'w') as file_for_pymol:
            file_for_pymol.write("""
            load  {s1}
            load  {s2}
            align {s3}, {s4}
            quit
            """.format(s1=structure_1, s2=structure_2,
                       s3=os.path.splitext(structure_1)[0],
                       s4=os.path.splitext(structure_2)[0]))
        # out_pymol = subprocess.check_output("module add pymol-1.8.2.1-gcc; pymol -c file_for_pymol.pml | grep Executive:; module rm pymol-1.8.2.1-gcc", shell=True)
        command = ' pymol -c {dir}/file_for_pymol.pml | grep Executive:'.format(dir = tmpdirname)
        out_pymol = subprocess.check_output(command, shell=True)
        rmsd = float(out_pymol[out_pymol.index(b'=') + 1:out_pymol.index(b'(') - 1])
    print('RMSD ', structure_1, ' and ', structure_2, ' = ', rmsd)


    return rmsd

def final_statistic(args):
    print(Colors.HEADER + 'FINAL STATISTIC \n' + Colors.ENDC)
    print('Number of repetitions', args.repeat, '\n')
    print('THE BEST RMSD')
    print('THE WORST RMSD')


def main():
    points = []
    args = get_argument()
    os.chdir(args.mydirvariable)
    list_pdb_file = find_pdb_file(args.mydirvariable)
    test_argument(args.n_files, args.k_options, list_pdb_file, args.tolerance)

    for i in range(args.repeat):
        c = RMSD_result()
        tmpdirname = tempfile.mkdtemp()
        print(Colors.OKGREEN + 'RUN {}/{}'.format(i + 1, args.repeat) + Colors.ENDC, '\n')
        all_files = random.sample(list_pdb_file, args.n_files)
        selected_files = random.sample(all_files, args.k_options)
        weights = np.random.dirichlet(np.ones(args.k_options), size=1)[0]
        files_and_weights = list(zip(selected_files, weights))
        c.selected_files = selected_files
        c.all_files = all_files
        c. files_and_weights = files_and_weights
        c.repeat = i+1
        points.append(c)

        if args.verbose:
            print_parameters_verbose(args, list_pdb_file, all_files, selected_files, files_and_weights)
        result_chi_and_weights = ensamble_fit(all_files, files_and_weights, tmpdirname)

        if args.k_options == 1:
            do_result(args.tolerance, all_files, selected_files, result_chi_and_weights, tmpdirname)
        else:
            print(Colors.WARNING +'not implemented \n' + Colors.ENDC)
        if not args.preserve:
            shutil.rmtree(tmpdirname)
    for point in points:
        point.print_result()
    final_statistic(args)




if __name__ == '__main__':
    main()
