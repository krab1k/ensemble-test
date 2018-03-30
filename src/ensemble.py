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
import logging
from time import localtime, strftime
from comparison import compare_ensembles
import threading


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

class LogPipe(threading.Thread):

    def __init__(self, level):
        """Setup the object with a logger and a loglevel
        and start the thread
        """
        threading.Thread.__init__(self)
        self.daemon = False
        self.level = level
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead)
        self.start()

    def fileno(self):
        """Return the write file descriptor of the pipe
        """
        return self.fdWrite

    def run(self):
        """Run the thread, logging everything.
        """
        for line in iter(self.pipeReader.readline, ''):
            logging.log(self.level, line.strip('\n'))

        self.pipeReader.close()

    def close(self):
        """Close the write end of the pipe.
        """
        os.close(self.fdWrite)

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
        if args.verbose == 3 or args.verbose == 2:
            print(Colors.OKBLUE + '\nSelected structure:\n' + Colors.ENDC)
            for structure, weight in list(zip(self.selected_files, self.weights)):
                print(f'structure: {structure} weight: {weight:.3f} \n')
        if args.verbose == 3 or args.verbose == 2:
            print(Colors.OKBLUE + '\nResults:\n ' + Colors.ENDC)
            for sumrmsd, chi2, data in self.results:
                print(f'RMSD: {sumrmsd:.3f} Chi2: {chi2:.3f}\n')
                for structure, weight in data:
                    print(f'structure: {structure} weight: {weight:.3f} \n')
        for sumrmsd, chi2, data in self.results:
            logging.info(f'###result_RMSD: {sumrmsd:5.3f} \n###result_CHI2: {chi2:5.3f}')
            for structure, weight in data:
                logging.info(f'#result_structure: {structure}| result_weight: {weight}')
    def get_best_result(self):
        return min(rmsd for rmsd, _, _ in self.results)

class SpecialFormatter(logging.Formatter):
    FORMATS = {logging.DEBUG : logging._STYLES['{'][0]("DEBUG: {message}"),
           logging.ERROR : logging._STYLES['{'][0]("{module} : {lineno}: {message}"),
           logging.INFO : logging._STYLES['{'][0]("{message}"),
           'DEFAULT' : logging._STYLES['{'][0](" {message}")}

    def format(self, record):
        # Ugly. Should be better
        self._style = self.FORMATS.get(record.levelno, self.FORMATS['DEFAULT'])
        return logging.Formatter.format(self, record)



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
    parser.add_argument("--verbose", type = int,
                        help="increase output verbosity, value 0, 1, 2, 3",
                        default=0)

    parser.add_argument("--verbose_logfile", help="increase output verbosity",
                       action="store_true")


    parser.add_argument("--tolerance", type=float, dest="tolerance",
                        help="pessimist (0) or optimist (0 < x <1) result",
                        default=0)

    parser.add_argument("--preserve", help="preserve temporary directory",
                        action="store_true")

    parser.add_argument("--method", help="choose method ensamble, eom, foxs",
                        choices=['ensemble', 'eom', 'multifoxs', 'mes'], required=True)


    parser.add_argument("--output", help="choose directory to save output",
                        metavar = "DIR", dest="output", required=True)


    parser.add_argument("--experimentdata", help="choose file for adderror",
                        metavar = "DIR", dest="experimentdata",required=True)


    return parser.parse_args()


def find_pdb_file(mydirvariable):
    pdb_files = []
    files = listdir(mydirvariable)
    for line in files:
        line = line.rstrip()
        if re.search('.pdb$', line):
            pdb_files.append(line)

    return pdb_files


def test_argument(args, list_pdb_file):
    if len(list_pdb_file) < args.n_files:
        print(Colors.WARNING + "Number of pdb files is ONLY" + Colors.ENDC, len(list_pdb_file), '\n')
        logging.error(f'Number of pdb files is ONLY {len(list_pdb_file)}')
        sys.exit(1)
    if args.k_options > args.n_files:
        print(Colors.WARNING + "Number of selected structure is ONLY" + Colors.ENDC, args.n_files, '\n')
        logging.error(f'Number of selected structure is ONLY { args.n_files}')
        sys.exit(1)
    if args.tolerance > 1:
        print('Tolerance should be less then 1.')
        logging.error('Tolerance should be less then 1.')
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
    print('-------------------------------------------')


def prepare_directory(all_files, tmpdir, method, verbose_logfile):
    pathlib.Path(tmpdir + '/pdbs').mkdir(parents=True, exist_ok=True)
    if method == 'ensemble':
        pathlib.Path(tmpdir + '/pdbs/ensembles').mkdir(parents=True, exist_ok=True)

    pathlib.Path(tmpdir + '/dats').mkdir(parents=True, exist_ok=True)
    pathlib.Path(tmpdir + '/method').mkdir(parents=True, exist_ok=True)
    pathlib.Path(tmpdir + '/results').mkdir(parents=True, exist_ok=True)
    # prepare 'file'.dat and copy to /dats/
    #logpipe = LogPipe(logging.INFO)
    for file in all_files:
        if verbose_logfile:
            logpipe = LogPipe(logging.DEBUG)
            # fox sends stdout to stderr by default
            logpipe_err = LogPipe(logging.DEBUG)
            return_value = subprocess.run(['foxs', f'{file}'], stdout = logpipe,
                                          stderr = logpipe_err)
            logpipe.close()
            logpipe_err.close()
        else:
            return_value = subprocess.run(['foxs', f'{file}'], stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
        if return_value.returncode:
            print(f'ERROR: Foxs failed.', file=sys.stderr)
            logging.error(f'Foxs failed.')
            sys.exit(1)

        shutil.copy(file + '.dat', f'{tmpdir}/dats/')

        if method == 'mes':
            dats_files = []
            files = listdir(tmpdir + '/dats/')
            for line in files:
                line = line.rstrip()
                if re.search('.dat$', line):
                    dats_files.append(line)
            lines = []
            for file in dats_files:
                with open(tmpdir + '/dats/' + file, 'w') as f:
                    f.writelines(lines[:0] + lines[2:])
                    shutil.copy(file, f'{tmpdir}/method/')

    if method == 'ensemble':  # format 01.pdb, 02.pdb as input for ensemble
        for i, f in enumerate(all_files, start=1):
            shutil.copy(f, f'{tmpdir}/pdbs/ensembles/{i:02d}.pdb')

    for file in all_files:  # not strict format for pdbs file
        shutil.copy(file, f'{tmpdir}/pdbs/')


def make_curve_for_experiment(files_and_weights, tmpdir, experimentdata):
    files = [filename for filename, weight in files_and_weights]
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

    adderror(experimentdata, tmpdir + '/method/curve')


def ensemble_fit(all_files, tmpdir, verbose, verbose_logfile):
    # RUN ensemble
    command = f'{ENSEMBLE_BINARY} -L -p {tmpdir}/pdbs/ensembles/ -n {len(all_files)} -m {tmpdir}/method/curve.modified.dat'
    if verbose == 3:
        print(Colors.OKBLUE + 'Command for ensemble fit \n' + Colors.ENDC, command, '\n')
    if verbose_logfile:
        logpipe = LogPipe(logging.DEBUG)
        logpipe_err = LogPipe(logging.ERROR)
        logging.info(f'Command for ensemble fit \n {command}')
        call = subprocess.run([f'{ENSEMBLE_BINARY}','-L','-p',f'{tmpdir}/pdbs/ensembles/','-n',f'{len(all_files)}',
                               '-m',f'{tmpdir}/method/curve.modified.dat'],
                              cwd=f'{tmpdir}/results/', stdout=logpipe, stderr=logpipe_err)
        logpipe.close()
        logpipe_err.close()
    else:
        call = subprocess.run(
            [f'{ENSEMBLE_BINARY}', '-L', '-p', f'{tmpdir}/pdbs/ensembles/', '-n', f'{len(all_files)}', '-m',
             f'{tmpdir}/method/curve.modified.dat'],
            cwd=f'{tmpdir}/results/', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if call.returncode:
        print(f'ERROR: ensemble failed', file=sys.stderr)
        logging.error(f'Ensemble failed.')

        sys.exit(1)
    # Process with result from ensemble
    result_chi_and_weights_ensemble = []
    # 5000
    # 1.08e+01,0.952,2.558,4.352610,0.000,0.300,0.000,0.000,0.000,0.000,0.000,0.092,0.000,0.908
    # 1.08e+01,0.950,2.558,4.752610,0.000,0.100,0.000,0.000,0.000,0.000,0.000,0.092,0.000,0.908
    with open(tmpdir + '/results/result', 'r') as f:
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


def multifoxs(all_files, tmpdir, verbose_logfile):
    # RUN Multi_foxs
    files_for_multifoxs = [str(tmpdir + '/pdbs/' + file) for file in all_files]
    if verbose_logfile:
        logpipe = LogPipe(logging.DEBUG)
        logpipe_err = LogPipe(logging.DEBUG)
        call = subprocess.run(['multi_foxs', f'{tmpdir}/method/curve.modified.dat',
                           *files_for_multifoxs], cwd=f'{tmpdir}/results/',
                          stdout=logpipe, stderr=logpipe_err)
        logpipe.close()
        logpipe_err.close()
    else:
        call = subprocess.run(['multi_foxs', f'{tmpdir}/method/curve.modified.dat',
                               *files_for_multifoxs], cwd=f'{tmpdir}/results/',
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if call.returncode: # multifoxs don't get right returnvalue
        print(f'ERROR: multifoxs failed', file=sys.stderr)
        logging.error(f'Multifoxs failed.')
        sys.exit(1)

    # Process with result from Multi_foxs
    multifoxs_files = []
    files = listdir(f'{tmpdir}/results/')
    for line in files:
        line = line.rstrip()
        if re.search('\d.txt$', line):
            multifoxs_files.append(line)
    result = []
    chi2 = 0
    for filename in multifoxs_files:
        with open(tmpdir + '/results/' + filename) as file:
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


def gajoe(all_files, tmpdir, verbose_logile):
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
    if verbose_logile:
        logpipe = LogPipe(logging.DEBUG)
        logpipe_err = LogPipe(logging.ERROR)
        p1 = subprocess.Popen(['yes'], stdout=subprocess.PIPE)
        call = subprocess.Popen(['gajoe', f'{tmpdir}/method/curve_gajoe.dat', f'-i={tmpdir}/method/juneom.eom',
                             '-t=5'], cwd=f'{tmpdir}/results/', stdin=p1.stdout,
                            stdout=logpipe, stderr=logpipe_err)
        call.communicate()
        logpipe.close()
        logpipe_err.close()
    else:
        p1 = subprocess.Popen(['yes'], stdout=subprocess.PIPE)
        call = subprocess.Popen(['gajoe', f'{tmpdir}/method/curve_gajoe.dat', f'-i={tmpdir}/method/juneom.eom',
                                 '-t=5'], cwd=f'{tmpdir}/results/', stdin=p1.stdout,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        call.communicate()
    if call.returncode:
        print(f'ERROR: GAJOE failed', file=sys.stderr)
        logging.error(f'GAJOE failed.')
        sys.exit(1)
    # process results from gajoe (/GAOO1/curve_1/
    chi2 = None
    structure_weight = []
    m = re.compile('^\s*\d+\)')
    with open(tmpdir + '/results/GA001/curve_1/logFile_001_1.log') as file_gajoe:
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

def mes(all_files, tmpdir, verbose_logfile):
    #Run MES

    # add empty line to curve,modified.dat
    with open(tmpdir + '/method/curve.modified.dat', 'r+') as f:
        line = ''
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
    # remove second line from file.dat, program accepts just one line at the beginning

    with open(tmpdir + '/method/filelist' , 'w') as file_mes:
        file_mes.write('curve.modified.dat' + '\n')
        for file in all_files:
            file_mes.write(file + '.dat' + '\n')
        # remove second line from saxs file
            call = subprocess.run(['sed' ,'-i','2d', f'{tmpdir}/method/' + f'{file}' + '.dat'])
            if call.returncode:
                print(f'ERROR: script failed', file=sys.stderr)
                logging.error(f'script failed.')
                sys.exit(1)


    logpipe = LogPipe(logging.DEBUG)
    logpipe_err = LogPipe(logging.ERROR)

    if verbose_logfile:
        with open(f'{tmpdir}/method/result_mes', 'a') as file_mes:
            call = subprocess.run(['/home/petra/Dokumenty/SAXS/mes/weights/mes', f'{tmpdir}/method/filelist'],
                                  cwd=f'{tmpdir}/method/',
                                  stdout = file_mes, stderr=logpipe_err)
    else:
        with open(f'{tmpdir}/method/result_mes', 'a') as file_mes:
            call = subprocess.run(['/home/petra/Dokumenty/SAXS/mes/weights/mes', f'{tmpdir}/method/filelist'],
                                  cwd=f'{tmpdir}/method/',
                                  stdout = file_mes, stderr=subprocess.PIPE)
    logpipe.close()
    logpipe_err.close()

    if call.returncode:
        print(f'ERROR: mes failed', file=sys.stderr)
        logging.error(f'mes failed.')
        sys.exit(1)

    result = []
    chi2 = 0
    weight_strucutre = []
    with open(tmpdir + '/method/result_mes') as file_mes:
        for line in file_mes:
            if line.startswith('  best xi'):
                chi2 = float(line.split(':')[1])
            if re.search('\d.pdb.dat', line):
                structure =  line.split('.')[0].strip() + '.pdb'
                weight = float(line.split(' ')[5].rstrip())
                weight_strucutre.append((structure, weight))
        result.append((chi2, weight_strucutre))
    return result

def process_result(tolerance, result_chi_structure_weights, run, tmpdir):
    minimum = min(chi2 for chi2, _ in result_chi_structure_weights)
    maximum = minimum * (1 + tolerance)

    all_results = []
    for chi2, names_and_weights in result_chi_structure_weights:
        result_files = [file for file, _ in names_and_weights]
        result_weights = [weight for _, weight in names_and_weights]
        if float(chi2) <= maximum:
            weighted_rmsd = compare_ensembles(run.selected_files, result_files, run.weights, result_weights)
            all_results.append((weighted_rmsd, chi2, names_and_weights))
    run.results = all_results
    return run


def final_statistic(runs, verbose):
    if verbose == 2 or verbose == 1 or verbose == 3:
        print('====================================================')
        print('====================================================')
        print(Colors.HEADER + '\nFINAL STATISTICS \n' + Colors.ENDC)
    rmsd = [result.get_best_result() for result in runs]
    if verbose == 2 or verbose == 1 or verbose == 3:
        print('Number of runs: ', len(runs))
    logging.info(f'*****All RMSDs| runs {len(runs)}:')
    for number in rmsd:
        logging.info(f'|{number:5.3f}|')

    indexes = [i for i, x in enumerate(rmsd) if x == min(rmsd)]
    if verbose == 2 or verbose == 1 or verbose == 3:
        print('Best RMSD {:5.3f}, run {}'.format(min(rmsd), *indexes))
        print('RMSD = {:.3f} ± {:.3f}'.format(np.mean(rmsd), np.std(rmsd)))
    logging.info('Best RMSD {:5.3f}, run {}'.format(min(rmsd), *indexes))
    logging.info(f'*****FINAL RMSD and STD| {np.mean(rmsd):5.3f}|{np.std(rmsd):5.3f}')

def main():
    random.seed(1)
    np.random.seed(1)
    args = get_argument()
    os.chdir(args.mydirvariable)
    list_pdb_file = find_pdb_file(args.mydirvariable)
    test_argument(args, list_pdb_file)
    result_chi_structure_weights = []
    all_runs = []
    hdlr = logging.FileHandler(f'{args.output}/result_{args.method}_n{args.n_files}_k{args.k_options}_{strftime("%Y-%m-%d__%H-%M-%S", localtime())}.log')
    hdlr.setFormatter(SpecialFormatter())
    logging.root.addHandler(hdlr)
    logging.root.setLevel(logging.INFO)
    logging.root.setLevel(logging.DEBUG)
    logging.info(f'***Output from ensemble*** {strftime("%Y-%m-%d__%H-%M-%S", localtime())} \n')
    logging.info(f'Assignment for experiment')
    logging.info(f'#Method: {args.method}')
    logging.info(f'#Repeats: {args.repeat}')
    logging.info(f'#All_files: {args.n_files}')
    logging.info(f'\n=============================\n')
    logging.info(f'An assignment for each iteration\n')
    logging.info(f'----------------------------------\n')
    if args.verbose == 3 or args.verbose == 2 or args.verbose == 1:
        print(f' \n EXPERIMENT  {strftime("%Y-%m-%d__%H-%M-%S", localtime())}')
    for i in range(args.repeat):
        tmpdir = tempfile.mkdtemp()
        logging.info(f'Task {i}')
        logging.info(f'#Working directory: {tmpdir}')
        if args.verbose ==  3 or args.verbose == 2:
            print('====================================================')
            print(Colors.OKGREEN + f'RUN {i+1}/{args.repeat} \n' + Colors.ENDC, '\n')
        all_files = random.sample(list_pdb_file, args.n_files)
        # copy to pds
        selected_files = random.sample(all_files, args.k_options)
        # copy to dats
        sample = np.random.dirichlet(np.ones(args.k_options), size=1)[0]
        weights = np.round(np.random.multinomial(1000, sample) / 1000, 3)
        files_and_weights = list(zip(selected_files,weights))
        logging.info(f'#Selected_files \n')
        for file1, weight1 in files_and_weights:
            logging.info(f'#structure {file1} | weight {weight1:5.3f}')
        # copy to methods
        prepare_directory(all_files, tmpdir, args.method, args.verbose_logfile)
        logging.info(f'\n==========================\n')
        make_curve_for_experiment(files_and_weights, tmpdir, args.experimentdata)
        if args.verbose == 3:
            print(Colors.OKBLUE + '\nCreated temporary directory \n' + Colors.ENDC, tmpdir, '\n')
            print(Colors.OKBLUE + 'Method' + Colors.ENDC, '\n')
            print(args.method, '\n')
        if args.verbose == 3:
            print_parameters_verbose(args, list_pdb_file, all_files)
        run = Run(all_files, selected_files, weights, i + 1, args.method)
        if args.method == 'ensemble':
            result_chi_structure_weights = ensemble_fit(all_files, tmpdir,
                                                        args.verbose, args.verbose_logfile)

        elif args.method == 'eom':
            result_chi_structure_weights = gajoe(all_files, tmpdir, args.verbose_logfile)

        elif args.method == 'multifoxs':
            result_chi_structure_weights = multifoxs(all_files, tmpdir, args.verbose_logfile)

        elif args.method == 'mes':
            result_chi_structure_weights = mes(all_files, tmpdir, args.verbose_logfile)

        run = process_result(args.tolerance, result_chi_structure_weights, run, tmpdir)

        all_runs.append(run)

        if not args.preserve:
            shutil.rmtree(tmpdir)

    for run in all_runs:
        run.print_result(args)
        logging.info(f'\n=============================\n')
    final_statistic(all_runs, args.verbose)


if __name__ == '__main__':
    main()
