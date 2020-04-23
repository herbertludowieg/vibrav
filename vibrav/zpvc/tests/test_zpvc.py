from vibrav.zpvc import ZPVC
from vibrav.base import resource
import numpy as np
import pandas as pd
import pytest
import tarfile
import os
import glob

@pytest.fixture
def zpvc_results():
    df = pd.read_csv(resource('nitromalonamide-zpvc-results.csv.xz'), compression='xz',
                     index_col=False)
    zpvc_results = df.groupby('temp')
    yield zpvc_results

@pytest.fixture
def zpvc_geometry():
    df = pd.read_csv(resource('nitromalonamide-zpvc-geometry.csv.xz'), compression='xz',
                     index_col=False)
    zpvc_geometry = df.groupby('temp')
    yield zpvc_geometry

@pytest.fixture
def grad(nat):
    df = pd.read_csv(resource('nitromalonamide-zpvc-grad.dat.xz'), compression='xz',
                     index_col=False, header=None)
    tmp = df.values.reshape(nat*((nat*3-6)*2+1), 3).T
    grad = pd.DataFrame.from_dict({'fx': tmp[0], 'fy': tmp[1], 'fz': tmp[2]})
    grad['file'] = np.repeat(range((nat*3-6)*2+1), nat)
    yield grad

@pytest.fixture
def prop():
    df = pd.read_csv(resource('nitromalonamide-zpvc-prop.dat.xz'), compression='xz',
                     index_col=False, header=None)
    df['file'] = df.index
    prop = df.copy()
    yield prop

@pytest.mark.parametrize("temp, nat", [([0], 15), ([100], 15), ([200], 15), ([300], 15),
                                       ([400], 15), ([600], 15)])
def test_zpvc(zpvc_results, zpvc_geometry, grad, prop, temp):
    with tarfile.open(resource('nitromalonamide-zpvc-dat-files.tar.xz'), 'r:xz') as tar:
        tar.extractall()
    zpvc = ZPVC(config_file=resource('nitromalonamide-zpvc-config.conf'))
    zpvc.zpvc(gradient=grad, property=prop, temperature=temp)
    data_files = glob.glob('*.dat')
    for file in data_files: os.remove(file)
    test_cols = ['tot_anharm', 'tot_curva', 'zpvc', 'property', 'zpva']
    exp_cols = ['anharm' ,'curv' ,'zpvc' ,'prop' ,'zpva']
    print(zpvc.vib_average.to_string())
    print(zpvc_results.get_group(temp[0])[exp_cols].values)
    print(zpvc.zpvc_results[test_cols].values)
    assert np.allclose(zpvc_results.get_group(temp[0])[exp_cols].values,
                       zpvc.zpvc_results[test_cols].values, atol=1e-3)

