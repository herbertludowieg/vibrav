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
from vibrav.numerical import vibronic_func
from vibrav.base import resource
from vibrav.util.io import open_txt
import pandas as pd
import numpy as np

def test_sf_to_so():
    spin_mult = [2, 1]
    states = [2, 3]
    nstates = 0
    nstates_sf = sum(states)
    multiplicity = []
    for mult, num_state in zip(spin_mult, states):
        multiplicity.append(np.repeat(mult, num_state))
        nstates += mult*num_state
    nstates = int(nstates)
    nstates_sf = int(nstates_sf)
    multiplicity = np.concatenate(tuple(multiplicity))
    # make test arrays
    sf = np.array([[0.0712009, 0.8316750, 0.8793323, 0.6863270, 0.7775856],
                   [0.7315691, 0.1706002, 0.3841723, 0.4575561, 0.1822313],
                   [0.3006358, 0.9992732, 0.5166593, 0.7038589, 0.7927969],
                   [0.8326891, 0.7536753, 0.6157563, 0.1457951, 0.9363394],
                   [0.6860063, 0.7598051, 0.5230968, 0.6029590, 0.5113386]])
    so = np.array([[0.0712009,         0, 0.8316750,         0,         0,         0,         0],
                   [        0, 0.0712009,         0, 0.8316750,         0,         0,         0],
                   [0.7315691,         0, 0.1706002,         0,         0,         0,         0],
                   [        0, 0.7315691,         0, 0.1706002,         0,         0,         0],
                   [        0,         0,         0,         0, 0.5166593, 0.7038589, 0.7927969],
                   [        0,         0,         0,         0, 0.6157563, 0.1457951, 0.9363394],
                   [        0,         0,         0,         0, 0.5230968, 0.6029590, 0.5113386]])
    extended = np.zeros((nstates, nstates), dtype=np.float64)
    vibronic_func.sf_to_so(nstates_sf, nstates, multiplicity, sf, extended)
    assert np.all(np.logical_not(np.isnan(extended)))
    assert np.allclose(extended, so)

def test_compute_d_dq():
    # setup
    spin_mult = [3, 1]
    states = [42, 49]
    nstates = 0
    nstates_sf = sum(states)
    multiplicity = []
    for mult, num_state in zip(spin_mult, states):
        multiplicity.append(np.repeat(mult, num_state))
        nstates += mult*num_state
    nstates = int(nstates)
    nstates_sf = int(nstates_sf)
    multiplicity = np.concatenate(tuple(multiplicity))
    # read data
    sf_dipoles = pd.read_csv(resource('molcas-ucl6-2minus-sf-dipole-1.txt.xz'), compression='xz',
                             header=0, index_col=False).values.reshape(nstates_sf, nstates_sf)
    so_dipoles = open_txt(resource('molcas-ucl6-2minus-so-dipole-1.txt.xz'), compression='xz').values
    eigvectors = open_txt(resource('molcas-ucl6-2minus-eigvectors.txt.xz'), compression='xz').values
    # allocate numpy arrays
    extended = np.zeros((nstates, nstates), dtype=np.float64)
    test_dipoles = np.zeros((nstates, nstates), dtype=np.complex128)
    # execute calculation
    vibronic_func.sf_to_so(nstates_sf, nstates, multiplicity, sf_dipoles, extended)
    vibronic_func.compute_d_dq(nstates, eigvectors, extended, test_dipoles)
    # assertions
    assert np.all(np.logical_not(np.isnan(test_dipoles)))
    assert np.allclose(test_dipoles, so_dipoles)

