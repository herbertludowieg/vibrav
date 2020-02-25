from vibrav.vibronic import Vibronic
from vibrav.base import resource
from vibrav.util.open_files import open_txt
import numpy as np
import pandas as pd
import tarfile
import os
import shutil

def test_vibronic_coupling():
    with tarfile.open(resource('molcas-ucl6-2minus-vibronic-coupling.tar.xz'), 'r:xz') as tar:
        tar.extractall()
    parent = os.getcwd()
    os.chdir('molcas-ucl6-2minus-vibronic-coupling')
    vib = Vibronic(config_file='va.conf')
    vib.vibronic_coupling(property='electric_dipole', print_stdout=False, temp=298,
                          write_property=False, write_oscil=True, sparse=True,
                          write_energy=False, verbose=False, eq_cont=False, select_fdx=0)
    base_oscil = open_txt(resource('molcas-ucl6-2minus-oscillators.txt.xz'), compression='xz',
                          rearrange=False)
    test_oscil = open_txt(os.path.join('vibronic-outputs', 'oscillators.txt'), rearrange=False)
    os.chdir(parent)
    shutil.rmtree('molcas-ucl6-2minus-vibronic-coupling')
    cols = ['nrow', 'ncol', 'oscil', 'energy']
    assert np.allclose(base_oscil.groupby('freqdx').get_group(0)[cols].values,
                       test_oscil[cols].values)

