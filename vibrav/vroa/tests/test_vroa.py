from vibrav.vroa import VROA
from vibrav.base import resource
import numpy as np
import pandas as pd
import tarfile
import os
import shutil


def test_vroa():
    with tarfile.open(resource('nwchem-h2o2-vroa.tar.gz'), 'r:gz') as tar:
        tar.extractall()
    parent = os.getcwd()
    os.chdir('nwchem-h2o2-vroa')
    cls = VROA(config_file='va.conf')
    cls.vroa(atomic_units=True)
    base_scatter = pd.read_csv('final-scatter.csv', index_col=False)
    cols = ['backscatter', 'forwardscatter']
    test = cls.scatter.copy()
    assert np.allclose(base_scatter[cols[0]], test[cols[0]])
    assert np.allclose(base_scatter[cols[1]], test[cols[1]])
    os.chdir(parent)
    shutil.rmtree('nwchem-h2o2-vroa')

