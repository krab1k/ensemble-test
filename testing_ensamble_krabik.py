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

"""ENSAMBLE, -d directory -n number of models """
"""-k number of selected structure"""
"""-r repet of program"""


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
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")

    parser.add_argument("-result", type=float, dest="result", help="pesimist(0) or optimist(<0) result",
                        default=0)

    return parser.parse_args()


def find_pdb_file(mydirvariable):
    pdb_files = []
    files = listdir(mydirvariable)
    for line in files:
        line = line.rstrip()
        if re.search('.pdb$', line):
            pdb_files.append(line)

    return pdb_files


def test_argument(n_files,k_options, list_pdb_file):
    if len(list_pdb_file) < n_files:
        print("Number od pdb files is ONLY", len(list_pdb_file))
        sys.exit(0)
    if k_options > n_files:
        print("Number of selected structure is ONLY", args.n_files)
        sys.exit(0)


def print_parametrs_verbose(n_files, k_options, list_pdb_file):
    print('Parametrs ')
    print ('Working directory', os.getcwd())
    print('Total number of pdb files is', len(list_pdb_file))
    print('The number of the selected files',
          n_files)
    print('The number of selected options', k_options)
    print('All pdb.dat files \n', list_pdb_file)

def ensamble_fit(k_options, n_files, selected_files_for_ensamble, data_for_experiment_modified):
    with tempfile.TemporaryDirectory(dir='.') as tmpdirname:
        print('created temporary directory', tmpdirname)
        i = 1
        for f in selected_files_for_ensamble:
            shutil.copy(f, tmpdirname + '/' + str(i).zfill(2) + '.pdb' )
            i += 1

        print('zkouska',tmpdirname[2:])
        name = input("Enter your name: ")   # Python 3
        curve_for_ensamble = make_curve_for_experiment(data_for_experiment_modified, k_options)
        command = '/storage/brno3-cerit/home/krab1k/saxs-ensamble-fit/core/ensamble-fit -L -p {path}{pdbdir}/ -n {n} -m {saxscurve}'.format(path = os.getcwd(), pdbdir=tmpdirname[1:], n=n_files, saxscurve=curve_for_ensamble)
        print(command)
        subprocess.call(command,shell=True)
    return ()


def work_with_result_from_ensamble():
    result_q_and_value = []
    with open('result', 'r') as f:
        next(f)
        # print (f.readline[2:4])
        for line in f:
            line = line.rstrip()
            print(line)
            value_of_chi2 = line.split(',')[3]
            values_of_index_result = line.split(',')[4:]
            result_q_and_value.append((value_of_chi2, values_of_index_result))

    print('value chi',value_of_chi2)
    print('index',values_of_index_result)
    print('result_q_and_value',result_q_and_value)

    return result_q_and_value


def do_result(result, select_random_files_for_experiment, data_for_experiment, f):
    result_q_and_value = work_with_result_from_ensamble()
    list_of_tuples = []
    if result == 0:
        tolerance = 0
    else:
        tolerance = float(result)
        if tolerance > 1:
            print('Less then 1')
            sys.exit(0)
    maximum = float(max(result_q_and_value)[0])

    minimum = maximum - maximum * tolerance
#    print(maximum, minimum)
#    print("result_q_and_value :")
#    print(result_q_and_value)
    for i, j in result_q_and_value:
        print('pocet select',len(select_random_files_for_experiment))
        print('delka j',len(j))
        if float(i) >= minimum:
            f.write('minimum a maximum:' + '\t' + str(minimum) + str(maximum) + '\n')
            for k in range(len(j)):
                if float(j[k]) != 0:
                    list_of_tuples.append((i, j[k], select_random_files_for_experiment[k]))
            print('vysledek', list_of_tuples)
            sum_rmsd = 0
            for k in list_of_tuples:
                sum_rmsd = sum_rmsd + rmsd_pymol(data_for_experiment[0], k[2], f) * float(k[1])
            f.write('sum rmsd' + '\t' + str(sum_rmsd) + '\n')
            list_of_tuples = []
            print('rmsd pymol', sum_rmsd)

    print(tolerance)

def make_curve_for_experiment(data_for_experiment_modified, k_options):
    print('data', data_for_experiment_modified)
    i = 0
    file_and_weight = []
    tmp = np.random.dirichlet(np.ones(k_options), size=1).transpose()
    for files in data_for_experiment_modified:
        file_and_weight.append((files, float(tmp[i])))
        i += 1
    print(file_and_weight)
    # result = []
    files = [open(file, 'r') for file in data_for_experiment_modified]
    with open('result.pdb.dat', 'w') as f:
        run = True
        while run:
            sum_result = 0
            q = 0
            i = 0
            for file in files:
                line = file.readline()
                if line == '':
                    run = False
                    break
                if not line.startswith('#'):
                    sum_result += float(line.split(' ')[4]) * file_and_weight[i][1]
                    q = float(line[:10])
                    #print(q)
            if sum_result != 0:
                # print(sum_result)
                f.write(str(q) + '\t' + str(sum_result) + '\t' + str(0) + '\n')
    for file in files:
        file.close()
    curve_for_ensamble = adderror("exp.dat", 'result.pdb.dat')
    # print(curve_for_ensamble)
    return curve_for_ensamble


def rmsd_pymol(structure_1, structure_2, f):
    if structure_1 == structure_2:
        print('double')
        rmsd = 0
    else:
        with open("file_for_pymol.pml", "w") as file_for_pymol:
            file_for_pymol.write("""
            load  {s1}
            load  {s2}
            align {s3}, {s4}
            quit
            """.format(s1=structure_1, s2=structure_2,
                       s3=os.path.splitext(structure_1)[0],
                       s4=os.path.splitext(structure_2)[0]))
        # part for home:
        # part for META:
        out_pymol = subprocess.check_output("module add pymol-1.8.2.1-gcc; pymol -c file_for_pymol.pml | grep Executive:; module rm pymol-1.8.2.1-gcc", shell=True)
        #out_pymol = subprocess.check_output(" pymol -c file_for_pymol.pml | grep Executive:", shell=True)
        rmsd = float(out_pymol[out_pymol.index(b'=') + 1:out_pymol.index(b'(') - 1])
        f.write('RMSD ' + '\t' + structure_1 + ' and ' + structure_2 + ' = ' + str(rmsd) + '\n')
        print('RMSD ', structure_1, ' and ', structure_2, ' = ', rmsd)
    return rmsd


def main():
    args = get_argument()
    os.chdir(args.mydirvariable)
    global list_pdb_file
    list_pdb_file = find_pdb_file(args.mydirvariable)
    test_argument(args.n_files,args.k_options, list_pdb_file)

    if args.verbose:
        print_parametrs_verbose(args.n_files, args.k_options, list_pdb_file)
    for i in range(args.repeat):
        filename = 'output_' + str(i) + str('_') + str(args.n_files)
        with open(filename, 'w') as f:
            select_random_files_for_experiment = random.sample(list_pdb_file, args.n_files)
            print(select_random_files_for_experiment)
            data_for_experiment = random.sample(select_random_files_for_experiment, args.k_options)
            f.write('N selected file' + str(select_random_files_for_experiment) + '\n')
            f.write('k options' + str(data_for_experiment) + '\n')
            data_for_experiment_modified = [None] * args.k_options
            for j in range(args.k_options):
                data_for_experiment_modified[j] = str(data_for_experiment[j]) + ".dat"
            ensamble_fit(args.k_options, args.n_files, select_random_files_for_experiment, data_for_experiment_modified)
            # work_with_result_from_ensamble( args)
            if args.k_options == 1:
                do_result(args.result, select_random_files_for_experiment, data_for_experiment, f)
            else:
                print('not implemented')
                sys.exit(0)

if __name__ == '__main__':
    main()
