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
from vibrav.util.io import open_txt
import pandas as pd

def block_diagonal(paths, out_file=None, pad=False):
    '''
    Combine multiple files in a txt format into a block
    diagonal matrix.
    '''
    dfs = []
    nrow_count = 0
    ncol_count = 0
    for path in paths:
        df = open_txt(path, rearrange=False)
        nrow_max = df['nrow'].unique().max() + 1
        ncol_max = df['ncol'].unique().max() + 1
        df[['nrow', 'ncol']] += [nrow_count, ncol_count]
        nrow_count += nrow_max
        ncol_count += ncol_max
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df

