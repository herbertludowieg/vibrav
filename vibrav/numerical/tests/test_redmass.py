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

