import sys
import os
import logging
try:
    import extpar.lib.utilities as utils
except ImportError:  # package not installed -> use PYTHONPATH
    import utilities as utils
'''
Module environment provides functions that interact with
the system Extpar is running on, it contains:

-get_cdo_version: get CDO version from command line `cdo -V`

-check_environment_for_extpar: check if all required modules are loaded

-get_omp_num_threads: get value of environment variable for OpenMP

-check_hdf5_threadsafe: checks if HDF5 compilation is threadsafe
'''


def get_cdo_version(extpar_programme, host):
    '''
    get CDO version from `cdo -V`

    If a CDO executable is not present in your PATH this
    will fail, and `launch_shell` will handle the error
    '''

    output = utils.launch_shell('cdo', '-V')
    cdo_version = output.split()[4]

    return cdo_version


def check_environment_for_extpar(extpar_programme):
    '''
    get hostname, python version, pythonpath and cdo version

    put all together into an info print for the logfile
    if CDO is not loaded, exit
    '''

    hostname = os.uname()[1]
    python_version = sys.version
    pythonpath = os.environ['PYTHONPATH']
    cdo_version = get_cdo_version(extpar_programme, hostname)

    logging.info('')
    logging.info('============= listen to environment ============')
    logging.info('')
    logging.info(f'Hostname -> {hostname}')
    logging.info('')
    logging.info(f'Python version -> {python_version}')
    logging.info('')
    logging.info(f'PYTHONPATH -> {pythonpath}')
    logging.info('')
    logging.info(f'CDO version -> {cdo_version}')
    logging.info('')


def get_omp_num_threads():
    '''
    get environment variables for OMP,
    if not set, assume 1 as default
    '''

    try:
        omp = os.environ['OMP_NUM_THREADS']
    except KeyError:
        omp = 1
        logging.warning('OMP_NUM_THREADS not set -> '
                        f'use OMP_NUM_THREADS = {omp} instead')
    return omp


def check_hdf5_threadsafe():
    '''
    Check if HDF5 compliation is threadsafe, otherwise return
    -L, which prevents crash when running cdo
    '''

    output = utils.launch_shell('cdo', '-V')
    if 'HDF5/threadsafe' in output:
        threadsafe = ''
    else:
        threadsafe = '-L'
        logging.info('HDF5 in CDO is not threadsafe -> add -L')
    return threadsafe
