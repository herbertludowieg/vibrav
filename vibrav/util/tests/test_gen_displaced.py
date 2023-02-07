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
from vibrav.util.gen_displaced import gen_delta, Displace
from vibrav.base import resource
import pytest
import numpy as np
import pandas as pd
import glob

params = ([1, 0.02, 'nien3-frequency-data.csv.xz', 'nien3-1-0.02-delta.dat.xz'],
          [1, 0.04, 'nien3-frequency-data.csv.xz', 'nien3-1-0.04-delta.dat.xz'],
          [1, 0.08, 'nien3-frequency-data.csv.xz', 'nien3-1-0.08-delta.dat.xz'],
          [2, 0.02, 'nien3-frequency-data.csv.xz', 'nien3-2-0.02-delta.dat.xz'],
          [2, 0.04, 'nien3-frequency-data.csv.xz', 'nien3-2-0.04-delta.dat.xz'],
          [2, 0.08, 'nien3-frequency-data.csv.xz', 'nien3-2-0.08-delta.dat.xz'])

@pytest.mark.parametrize("delta_type, norm, freq, expected", params)
def test_gen_delta(delta_type, norm, freq, expected):
    freq_df = pd.read_csv(resource(freq), compression='xz')
    exp_df = pd.read_csv(resource(expected), header=None, compression='xz').values.reshape(-1,)
    delta = gen_delta(freq=freq_df, delta_type=delta_type, norm=norm)
    assert np.allclose(exp_df, delta['delta'].values.reshape(-1,))

def test_normal_modes():
    def read_file(fp):
        return pd.read_csv(fp, index_col=False, header=None).values.flatten()
    disp = Displace(cls=None, write_files=False, atom_file=resource('h2-atom.csv'),
                    freq_file=resource('h2-frequency.csv'), csv_props=True,
                    norm=[0.02, 0.04])
    actual = pd.read_csv(resource('h2-norm-disps.csv'), index_col=0, header=0)
    cols = ['x', 'y', 'z']
    assert np.allclose(disp.disp[cols], actual[cols])
    files = glob.glob('*.dat')
    for f in files:
        test = read_file(f)
        actual = read_file(resource('h2-'+f))
        assert np.allclose(test, actual)

def test_cartestian():
    disp = Displace(cls=None, write_files=False, atom_file=resource('h2-atom.csv'),
                    freq_file=resource('h2-frequency.csv'), csv_props=True,
                    cart_disp=True, disp=0.01)
    actual = pd.read_csv(resource('h2-cart-disps.csv'), index_col=0, header=0)
    cols = ['x', 'y', 'z']
    assert np.allclose(disp.disp[cols], actual[cols])

