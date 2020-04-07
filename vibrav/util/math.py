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

def get_triu(arr, k=0):
    '''
    Get the upper triangular indeces of the input matrix

    Args:
        arr (:obj:`numpy.array`): Array to parse the upper triangular elements
        k (:obj:`int`): k parameter that goes into the `np.triu_indices_from` function. Refer to numpy
                        documentation for more information.

    Returns:
        triu_arr (:obj:`numpy.array`): 1D array with the upper triangular elements
    '''
    # get the elements in the upper triangular
    triu = np.triu_indices_from(arr, k=k)
    # return the elements as a 1d array
    triu_arr = arr[triu]
    return triu_arr

def ishermitian(data):
    '''
    Check if the input array is hermitian.

    Note:
        This function does not determine if there are any non-numeric values.
        It assumes that you are feeding an array of floats, ints, etc.

    Args:
        data (:obj:`numpy.array`): Array to be evaluated

    Return:
        isherm (:obj:`bool`): Is the array hermitian
    '''
    herm = np.conjugate(np.transpose(data))
    isherm = np.allclose(herm, data)
    return isherm

