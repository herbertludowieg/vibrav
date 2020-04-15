import numpy as np
import pandas as pd

def energetic_degeneracy(data_df, degen_delta, rtol=1e-12, numpy=True):
    '''
    Get the energetic degeneracies within the energy tolerance given by the
    `degen_delta` parameter. This is mean to keep track of the indeces at which
    the energies are degenerate which helps when the input energies are not
    sorted in increasing order of energy. This is largely due to the way the
    function is set up.

    The returned data frame has the columns 'value', 'degen', and 'index' for
    the energies, electronic degeneracies, and indeces where the degenerate
    energies are based on the input array locations.

    If the input data is in 'Hartree' we recommend a degeneracy parameter of
    1e-5 Ha. For data that is in 'wavenumbers' we recommend a degeneracy
    parameter of 1 cm^{-1}

    Note:
        There is no kind of sanity checking with the tolerance values that are
        given. Meaning, that if the user gives an un-physical degeneracy
        tolerance parameter this function will blindly calculate it.

    Args:
        data_df (:obj:`pandas.DataFrame` or :obj:`numpy.ndarray`):
                    Data frame or array of the energies. If it is a numpy array
                    must set the `numpy` parameter to `True`.
        degen_delta (:obj:`float`): Absolute value for determining two levels
                                    are degenerate. Must be in the same units
                                    as the input energies.
        rtol (:obj:`float`, optional): Relative tolerance value for the
                                       differences in energy. Defaults to
                                       `1e-12` so the it is more dependent on
                                       the `degen_delta` parameter.
        numpy (:obj:`bool`, optional): Tell the program that the input data is
                                       a numpy array instead of a pandas data
                                       frame. Defaults to `True`.
    Returns:
        degeneracy (:obj:`pandas.DataFrame`): Data frame containing the
                                              degenerate energies.
    '''
    degen_states = []
    idx = 0
    # convert to a pandas series object
    # we then sort by the energies without reseting the index
    # because that way we keep track of the input energy ordering
    if not numpy:
        sorted = data_df.sort_values()
        index = sorted.index.values
        data = sorted.values
    else:
        df = pd.Series(data_df)
        sorted = df.sort_values()
        index = sorted.index.values
        data = sorted.values
    # iterate over all of the energies
    while idx < data.shape[0]:
        # determine what is degenerate
        degen = np.isclose(data[idx], data, atol=degen_delta, rtol=rtol)
        # get the locations
        ddx = np.where(degen)[0]
        # get the energies and indeces
        degen_vals = data[ddx]
        degen_index = index[ddx]
        mean = np.mean(degen_vals)
        # add however many degenerate energies are found
        idx += ddx.shape[0]
        # put everything together
        df = pd.DataFrame.from_dict({'value': [mean], 'degen': [ddx.shape[0]]})
        found = np.transpose(degen_index)
        df['index'] = [found]
        degen_states.append(df)
    degeneracy = pd.concat(degen_states, ignore_index=True)
    return degeneracy

