from vibrav.util.data_checking import check_size, NaNValueFound, SizeError
import pandas as pd
import numpy as np
import pytest

df = pd.DataFrame(np.random.rand(3,3))
arr = np.random.rand(3,3)
to_fail_arr = np.random.rand(10,10)
to_fail_df = pd.DataFrame(np.random.rand(10,10))

@pytest.mark.parametrize('data,dataframe,fail',
                         [(df, True, False), (arr, False, False),
                          (to_fail_arr, False, True),
                          (to_fail_df, True, True)])
def test_data_checking(data, dataframe, fail):
    if not fail:
        check_size(data, (3,3), 'test', dataframe)
    else:
        for idx in np.random.randint(0,10,3):
            for jdx in np.random.randint(0,10,3):
                data[idx][jdx] = np.NAN
        with pytest.raises(SizeError):
            check_size(data, (3,3), 'test', dataframe)
        with pytest.raises(NaNValueFound):
            check_size(data, (10,10), 'test', dataframe)
