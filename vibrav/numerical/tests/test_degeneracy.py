from vibrav.numerical.degeneracy import energetic_degeneracy
from vibrav.base import resource
import numpy as np
import pandas as pd

def test_energetic_degeneracy():
    degen = pd.read_csv(resource('molcas-rassi-nien-degen-so-energy.csv.xz'), compression='xz',
                        index_col=False)
    df = pd.read_csv(resource('molcas-rassi-nien-energy.csv.xz'), compression='xz', index_col=False)
    test_degen = energetic_degeneracy(df['so'].values, 1e-5)
    cols = ['value', 'degen']
    assert np.allclose(degen[cols].values, test_degen[cols].values)

