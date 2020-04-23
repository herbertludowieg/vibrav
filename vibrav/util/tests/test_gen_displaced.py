from vibrav.util.gen_displaced import gen_delta
from vibrav.base import resource
import pytest
import numpy as np
import pandas as pd

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
    
