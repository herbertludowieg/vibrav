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
from vibrav.numerical.redmass import rmass_mwc, rmass_cart
from vibrav.adf.output import Tape21
from vibrav.util.io import uncompress_file
from vibrav.base import resource
import pandas as pd
import numpy as np
import pytest
import os

file_params = [('adf-ubr-1minus-b3lyp-numeric-norm-modes-{}.csv.xz',
                'adf-ubr-1minus-b3lyp-numeric-redmass.txt.xz'),
               ('adf-ubr-1minus-pbe-numeric-norm-modes-{}.csv.xz',
                'adf-ubr-1minus-pbe-numeric-redmass.txt.xz')]

dalton2au = 1822.888484770052

@pytest.mark.parametrize('test_file, expected', file_params)
def test_rmass_mwc(test_file, expected):
    test_freqs = pd.read_csv(resource(test_file.format('mwc')), compression='xz',
                             index_col=False)
    symbols = ['U']+['Br']*6
    test_rmass = test_freqs.groupby('freqdx').apply(rmass_mwc, symbols)
    exp_rmass = pd.read_csv(resource(expected), compression='xz', comment='#', header=None,
                            index_col=False)
    exp_rmass = exp_rmass.values.reshape(-1,) / dalton2au
    assert np.allclose(exp_rmass, test_rmass)

@pytest.mark.parametrize('test_file, expected', file_params)
def test_rmass_cart(test_file, expected):
    test_freqs = pd.read_csv(resource(test_file.format('cart')), compression='xz',
                             index_col=False)
    symbols = ['U']+['Br']*6
    test_rmass = test_freqs.groupby('freqdx').apply(rmass_cart, symbols)
    exp_rmass = pd.read_csv(resource(expected), compression='xz', comment='#', header=None,
                            index_col=False)
    exp_rmass = exp_rmass.values.reshape(-1,) / dalton2au
    assert np.allclose(exp_rmass, test_rmass)

