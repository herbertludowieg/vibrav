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
from vibrav.util.block_diagonal import block_diagonal
from vibrav.base import resource
import pandas as pd
import numpy as np

def test_block_diagonal():
    df1 = pd.read_csv(resource('nichxn3-molcas-sing-ham-sf.txt.xz'),
                      index_col=False, delim_whitespace=True)
    df2 = pd.read_csv(resource('nichxn3-molcas-trip-ham-sf.txt.xz'),
                      index_col=False, delim_whitespace=True)
    cols = ['nrow', 'ncol', 'real', 'imag']
    df1.columns = cols
    df2.columns = cols
    paths = [resource('nichxn3-molcas-sing-ham-sf.txt.xz'),
             resource('nichxn3-molcas-trip-ham-sf.txt.xz')]
    df = block_diagonal(paths)
    sing_vals = df.groupby('nrow').filter(lambda x: x['nrow'].unique() \
                                                in range(45))['real'].values
    trip_vals = df.groupby('nrow').filter(lambda x: x['nrow'].unique() \
                                                in range(45, 85))['real'].values
    assert np.allclose(df1['real'].values, sing_vals)
    assert np.allclose(df2['real'].values, trip_vals)

