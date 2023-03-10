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

class NaNValueFound(Exception):
    pass

class SizeError(Exception):
    pass

def check_size(data, size, var_name, dataframe=False):
    '''
    Simple method to check the size of the input array.

    Args:
        data (np.array or pd.DataFrame): Data array to determine the size.
        size (tuple): Size expected.
        var_name (str): Name of the variable for printing in the error message.

    Raises:
        ValueError: If the given array does not have the shape given in the
            `size` parameter.
        TypeError: If there are any not a number or null values in the array.
    '''
    if data.shape != size:
        msg = "'{var}' is not of proper size, currently {curr} expected {ex}"
        raise SizeError(msg.format(var=var_name, curr=data.shape,
                                    ex=size))
    try:
        _ = np.any(np.isnan(data))
        numpy = True
    except TypeError:
        numpy = False
    msg = "NaN values were found in the data for '{}'"
    if numpy:
        if np.any(np.isnan(data)):
            raise NaNValueFound(msg.format(var_name))
    else:
        if np.any(pd.isnull(data)):
            raise NaNValueFound(msg.format(var_name))

