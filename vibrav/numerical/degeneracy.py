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
import numpy as np
import pandas as pd

def energetic_degeneracy(data_df, degen_delta, rtol=1e-12, numpy=True, original_order=False,
                         min_sort=False, original_size=False):
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
                    must set the `numpy` parameter to :code:`True`.
        degen_delta (:obj:`float`): Absolute value for determining two levels
                                    are degenerate. Must be in the same units
                                    as the input energies.
        rtol (:obj:`float`, optional): Relative tolerance value for the
                                       differences in energy. Defaults to
                                       :code:`1e-12` so the it is more dependent on
                                       the `degen_delta` parameter.
        numpy (:obj:`bool`, optional): Tell the program that the input data is
                                       a numpy array instead of a pandas data
                                       frame. Defaults to :code:`True`.
    Returns:
        degeneracy (:obj:`pandas.DataFrame`): Data frame containing the
                                              degenerate energies.
    '''
    degen_states = []
    original = []
    idx = 0
    # convert to a pandas series object
    # we then sort by the energies without reseting the index
    # because that way we keep track of the input energy ordering
    if not numpy:
        df = pd.DataFrame.from_dict({'energy': data_df.values.flatten(),
                                     'sort': range(data_df.shape[0])})
        sorted_vals = df.sort_values(by=df.columns[0])
        index = sorted_vals.index.values
        data = sorted_vals['energy'].values
    else:
        df = pd.DataFrame.from_dict({'energy': data_df, 'sort': range(data_df.shape[0])})
        sorted_vals = df.sort_values(by=df.columns[0])
        index = sorted_vals.index.values
        data = sorted_vals['energy'].values
    original_data = pd.DataFrame.from_dict({'energy': data})
    original_data.index = index
    original_data['index'] = 0
    original_data['degen'] = 0
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
        # put everything together
        if min_sort and False:
            df = pd.DataFrame.from_dict({'value': [mean], 'degen': [ddx.shape[0]],
                                         'sort': min(degen_index)})
        else:
            df = pd.DataFrame.from_dict({'value': [mean], 'degen': [ddx.shape[0]],
                                         'sort': index[idx]})
        idx += ddx.shape[0] if not original_size else 1
        found = np.transpose(degen_index)
        df['index'] = [found]
        #ndf = original_data.loc[ddx].copy()
        #ndf['degen'] = ddx.shape[0]
        #ndf['index'] = [found]*ddx.shape[0]
        #ndf['degen_energy'] = [mean]*ddx.shape[0]
        #for d in ddx:
        #    print(original_data.loc[d])
        #    print(d, found)
        #    original_data.loc[d, 'degen'] = ddx.shape[0]
        #    original_data.loc[d, 'index'] = [found]
        degen_states.append(df)
        #if original_size:
        #    original.append(ndf)
    degeneracy = pd.concat(degen_states, ignore_index=True)
    if original_size:
        degeneracy.index = index
    if original_order:
        degeneracy.sort_values(by=['sort'], inplace=True)
        degeneracy.reset_index(drop=True, inplace=True)
    if not original_size or True:
        return degeneracy
    else:
        original = pd.concat(original)
        return degeneracy, original


