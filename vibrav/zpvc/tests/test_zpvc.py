# This file is part of vibrav.
#
# vibrav is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# vibrav is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with vibrav.  If not, see <https://www.gnu.org/licenses/>.
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

# have to make a rtol input due to the test data accuracy
@pytest.mark.parametrize("temp, nat, rtol", [([  0], 15, 1e-4), ([100], 15, 1e-5),
                                        ([200], 15, 1e-5), ([300], 15, 1e-5),
                                        ([400], 15, 1e-5), ([600], 15, 1e-5)])
def test_zpvc(zpvc_results, zpvc_geometry, grad, prop, temp, nat, rtol):
    zpvc = ZPVC(config_file=resource('nitromalonamide-zpvc-config.conf'))
    zpvc.zpvc(gradient=grad, property=prop, temperature=temp, write_out_files=False)
    test_cols = ['tot_anharm', 'tot_curva', 'zpvc', 'property', 'zpva']
    exp_cols = ['anharm' ,'curv' ,'zpvc' ,'prop' ,'zpva']
    print("Test values")
    print(zpvc_results.get_group(temp[0])[exp_cols].values)
    print("Calculated values")
    print(zpvc.zpvc_results[test_cols].values)
    print("Comparison with np.isclose(atol=1e-3, rtol=1e-4)")
    print(np.isclose(zpvc_results.get_group(temp[0])[exp_cols].values,
                       zpvc.zpvc_results[test_cols].values, atol=1e-3, rtol=1e-4))
    print("Comparison with np.isclose(atol=1e-3, rtol=1e-5)")
    print(np.isclose(zpvc_results.get_group(temp[0])[exp_cols].values,
                       zpvc.zpvc_results[test_cols].values, atol=1e-3, rtol=1e-5))
    assert np.allclose(zpvc_results.get_group(temp[0])[exp_cols].values,
                       zpvc.zpvc_results[test_cols].values, atol=1e-3, rtol=rtol)

