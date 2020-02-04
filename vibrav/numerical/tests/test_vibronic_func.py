from vibrav.numerical import vibronic_func
from vibrav.base import resource
from vibrav.util.open_files import open_txt
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
    sf = np.array([[0.07120099, 0.83167500, 0.87933230, 0.68632707, 0.77758568],
                   [0.73156913, 0.17060028, 0.38417238, 0.45755610, 0.18223135],
                   [0.30063586, 0.99927320, 0.51665930, 0.70385890, 0.79279694],
                   [0.83268917, 0.75367538, 0.61575638, 0.14579517, 0.93633949],
                   [0.68600632, 0.75980517, 0.52309685, 0.60295909, 0.51133868]])
    so = np.array([[0.07120099,          0, 0.83167500,          0,          0,          0,          0],
                   [         0, 0.07120099,          0, 0.83167500,          0,          0,          0],
                   [0.73156913,          0, 0.17060028,          0,          0,          0,          0],
                   [         0, 0.73156913,          0, 0.17060028,          0,          0,          0],
                   [         0,          0,          0,          0, 0.51665930, 0.70385890, 0.79279694],
                   [         0,          0,          0,          0, 0.61575638, 0.14579517, 0.93633949],
                   [         0,          0,          0,          0, 0.52309685, 0.60295909, 0.51133868]])
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
    print(sf_dipoles.dtype, so_dipoles.dtype, eigvectors.dtype, multiplicity.dtype)
    print(type(nstates), type(nstates_sf))
    # allocate numpy arrays
    extended = np.zeros((nstates, nstates), dtype=np.float64)
    test_dipoles = np.zeros((nstates, nstates), dtype=np.complex128)
    # execute calculation
    vibronic_func.sf_to_so(nstates_sf, nstates, multiplicity, sf_dipoles, extended)
    vibronic_func.compute_d_dq(nstates, eigvectors, extended, test_dipoles)
    # assertions
    assert np.all(np.logical_not(np.isnan(test_dipoles)))
    assert np.allclose(test_dipoles, so_dipoles)

