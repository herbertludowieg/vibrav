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
import pandas as pd
import numpy as np

def open_txt(fp, rearrange=True, **kwargs):
    '''
    Method to open a .txt file that has a separator of ' ' with the first columns ordered as
    ['nrow', 'ncol', 'real', 'imag']. We take care of adding the two values to generate a
    complex number. The program will take your data and automatically generate a matrix
    of complex values that has the size of the maximum of the unique values in the 'nrow'
    column and the maximum in the 'ncol' column. tha being said if there are missing values
    the program will throw 'ValueError' as it will not be able to determine the size of the
    new matrix. This works for both square and non-square matrices. We assume that the indexing
    is non-pythonic hence the subtraction of 'nrow' and 'ncol' columns.

    Args:
        fp (str): Filepath of the file you want to open.
        sep (str, optional): Delimiter value.
        skipinitialspace (bool, optional): Pandas skipinitialspace argument in the `pandas.read_csv`
                                           method.
        rearrange (bool, optional): If you want to rearrange the data into a square complex matrix

    Returns:
        matrix (pandas.DataFrame): Re-sized complex square matrix with the appropriate size.

    Raises:
        TypeError: When there are null values found. A common place this has been an issue is when
                   the first two columns in the '.txt' files read have merged due to too many states.
                   This was found to happen when there were over 1000 spin-orbit states.
    '''
    keys = kwargs.keys()
    if 'skipinitialspace' not in keys:
        kwargs['skipinitialspace'] = True
    if 'sep' not in keys:
        kwargs['sep'] = ' '
    if 'index_col' not in keys:
        kwargs['index_col'] = False
    df = pd.read_csv(fp, **kwargs)
    if pd.isnull(df).values.any():
        print(np.where(pd.isnull(df)))
        raise TypeError("Null values where found while reading {fp} ".format(fp=fp) \
                       +"a common issue is if the ncol and nrow values have merged")
    # this is an assumption that only works in the context of this program as the
    # txt files that the program is looking for have four columns
    # TODO: might be nice to give a conditional to check if the data is real or complex
    #       and only read the first three columns but that might be more work for nothing
    df.columns = list(map(lambda x: x.lower().replace('#', ''), df.columns))
    df['nrow'] -= 1
    df['ncol'] -= 1
    # rearrange the data to a matrix ordered by the rows and columns
    if rearrange:
        matrix = pd.DataFrame(df.groupby('nrow').apply(lambda x:
                              x['real']+1j*x['imag']).values.reshape(len(df['nrow'].unique()),
                                                                     len(df['ncol'].unique())))
    else:
        matrix = df.copy()
    return matrix

