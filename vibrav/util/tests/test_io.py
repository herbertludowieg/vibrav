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
from vibrav.util.io import read_data_file, open_txt
from vibrav.base import resource
import pandas as pd
import numpy as np
import pytest

@pytest.mark.parametrize('fp,nmodes,smat,nat,fail',
                         [(resource('h2o2-vroa-smatrix.dat.xz'), 6, True, 4, False),
                          (resource('h2o2-vroa-freq.dat.xz'), 6, False, None, False),
                          (resource('h2o2-vroa-freq.dat.xz'), 4, False, None, True)])
def test_read_data_file(fp, nmodes, smat, nat, fail):
    if not fail and not smat:
        actual = [342.372066, 857.146267, 1242.31013, 1371.35667,
                  3591.20078, 3591.92818]
        test = read_data_file(fp, nmodes, smat, nat)
        assert len(test) == 6
        assert np.allclose(test, actual)
    elif not fail and smat:
        actual = pd.read_csv(resource('h2o2-normal-modes.csv.bz2'),
                             index_col=0, header=0)
        test = read_data_file(fp, nmodes, smat, nat)
        assert test.shape == (24, 4)
        assert np.allclose(test, actual)
    else:
        with pytest.raises(ValueError):
            assert read_data_file(fp, nmodes, smat, nat)

def test_open_txt_real_square():
    test_file = resource('nichxn3-molcas-sing-ham-sf.txt.xz')
    test_df = open_txt(test_file, is_complex=False)
    assert test_df.shape == (45, 45)
    assert np.all(pd.notna(test_df))
    test_df = open_txt(test_file, rearrange=False, tol=1e-9)
    assert test_df.shape[1] == 4
    assert np.all(pd.notna(test_df))

def test_open_txt_real_non_square():
    test_file = resource('nichxn3-molcas-full-ham-sf.txt.xz')
    test_df = open_txt(test_file, is_complex=False, fill=True)
    assert test_df.shape == (85,85)
    assert np.all(pd.notna(test_df))
    with pytest.raises(ValueError):
        open_txt(test_file, is_complex=False)
    test_df = open_txt(test_file, rearrange=False, tol=1e-9)
    assert test_df.shape == (3625, 4)
    assert np.all(pd.notna(test_df))

def test_open_txt_comp_square():
    test_file = resource('ucl6-2minus-vibronic-eigvectors.txt.xz')
    test_df = open_txt(test_file)
    assert test_df.shape == (175, 175)
    assert np.all(pd.notna(test_df))
    with pytest.raises(ValueError):
        open_txt(test_file, is_complex=False)
    test_df = open_txt(test_file, rearrange=False, get_complex=True,
                       get_magnitude=True)
    assert test_df.shape == (30625, 6)
    assert np.all(pd.notna(test_df))
    assert np.allclose(test_df['complex'], test_df['real']+1j*test_df['imag'])
    assert np.allclose(test_df['magnitude'], np.sqrt(test_df['real']**2+test_df['imag']**2))

