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
from vibrav.vibronic import Vibronic
from vibrav.base import resource, list_resource
from vibrav.util.io import open_txt
import numpy as np
import pandas as pd
import pytest
import os

@pytest.mark.parametrize('freqdx', [[-1]])
def test_vibronic_coupling(freqdx):
    vib = Vibronic(config_file=resource('ucl6-2minus-vibronic-va.conf'))
    vib.vibronic_coupling(property='electric_dipole', print_stdout=False, temp=298,
                          write_property=False, write_oscil=True, boltz_states=2,
                          write_energy=False, verbose=False, eq_cont=False, select_fdx=freqdx)
    base_oscil = open_txt(resource('molcas-ucl6-2minus-oscillators.txt.xz'), compression='xz',
                          rearrange=False)
    test_oscil = open_txt(os.path.join('vibronic-outputs', 'oscillators-0.txt'), rearrange=False)
    test_oscil = test_oscil[np.logical_and(test_oscil['oscil'].values > 0,
                                           test_oscil['energy'].values > 0)]
    cols = ['oscil', 'energy']
    if freqdx[0] == -1:
        freqdx = range(15)
    base = base_oscil.groupby('freqdx').filter(lambda x: x['freqdx'].unique()
                                                         in freqdx)
    base.sort_values(by=['freqdx', 'sign', 'nrow', 'ncol'], inplace=True)
    base = base[cols].values
    test = test_oscil.groupby('sign').filter(lambda x: x['sign'].unique()
                                                       in ['minus', 'plus'])
    test.sort_values(by=['freqdx', 'sign', 'nrow', 'ncol'], inplace=True)
    test = test[cols].values
    assert np.allclose(base[:,0], test[:,0], rtol=7e-5)
    assert np.allclose(base[:,1], test[:,1], rtol=1e-5, atol=1e-7)

