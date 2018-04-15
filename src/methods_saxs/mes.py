# prepare_data()
# make_experiment()
# collect_result()
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
import pathlib
import logging
from time import localtime, strftime
from comparison import compare_ensembles
import threading


def prepare_data(all_files, tmpdir, method, verbose_logfile):
    for file in all_files:  # not strict format for pdbs file
        shutil.copy(file, f'{tmpdir}/pdbs/')


def make_experiment(all_files, tmpdir, verbose, verbose_logfile, method):
    # Run MES

    # add empty line to curve,modified.dat
    with open(tmpdir + '/method/curve.modified.dat', 'r+') as f:
        line = ''
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)
    # remove second line from file.dat, program accepts just one line at the beginning

    with open(tmpdir + '/method/filelist', 'w') as file_mes:
        file_mes.write('curve.modified.dat' + '\n')
        for file in all_files:
            file_mes.write(file + '.dat' + '\n')
            # remove second line from saxs file
            call = subprocess.run(['sed', '-i', '2d', f'{tmpdir}/method/' + f'{file}' + '.dat'])
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
                                  stdout=file_mes, stderr=logpipe_err)
    else:
        with open(f'{tmpdir}/method/result_mes', 'a') as file_mes:
            call = subprocess.run(['/home/petra/Dokumenty/SAXS/mes/weights/mes', f'{tmpdir}/method/filelist'],
                                  cwd=f'{tmpdir}/method/',
                                  stdout=file_mes, stderr=subprocess.PIPE)
    logpipe.close()
    logpipe_err.close()

    if call.returncode:
        print(f'ERROR: mes failed', file=sys.stderr)
        logging.error(f'mes failed.')
        sys.exit(1)


def collect_results(tmpdir):
    result = []
    chi2 = 0
    weight_strucutre = []
    with open(tmpdir + '/method/result_mes') as file_mes:
        for line in file_mes:
            if line.startswith('  best xi'):
                chi2 = float(line.split(':')[1])
            if re.search('\d.pdb.dat', line):
                structure = line.split('.')[0].strip() + '.pdb'
                weight = float(line.split(' ')[5].rstrip())
                weight_strucutre.append((structure, weight))
        result.append((chi2, weight_strucutre))
    return result
