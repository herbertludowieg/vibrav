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
from numba import vectorize, float64, complex128

def get_triu(arr, k=0):
    '''
    Get the upper triangular indeces of the input matrix

    Args:
        arr (:obj:`numpy.array`): Array to parse the upper triangular elements
        k (:obj:`int`): k parameter that goes into the `np.triu_indices_from` function. Refer to
                        numpy documentation for more information.

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

def isantihermitian(data):
    '''
    Check if the input array is symmetric.

    Note:
        This function does not determine if there are any non-numeric values.
        It assumes that you are feeding an array of floats, ints, etc.

    Args:
        data (:obj:`numpy.array`): Array to be evaluated

    Return:
        isherm (:obj:`bool`): Is the array hermitian
    '''
    if isinstance(data, (list, tuple)): data = np.array(data)
    a = -1*(np.ones(data.shape) - np.eye(data.shape[0])) + np.eye(data.shape[0])
    antiherm = a*np.conjugate(np.transpose(data))
    isantiherm = np.allclose(antiherm, data)
    return isantiherm

def issymmetric(data):
    '''
    Check if the input array is symmetric.

    Note:
        This function does not determine if there are any non-numeric values.
        It assumes that you are feeding an array of floats, ints, etc.

    Args:
        data (:obj:`numpy.array`): Array to be evaluated

    Return:
        isherm (:obj:`bool`): Is the array hermitian
    '''
    if np.iscomplex(data).any():
        raise TypeError("The data type that was detected in the passed data were " \
                        +"complex instead or real values. Use ishermitian instead.")
    symm = np.transpose(data)
    issymm = np.allclose(symm, data)
    return issymm

def isantisymmetric(data):
    '''
    Check if the input array is symmetric.

    Note:
        This function does not determine if there are any non-numeric values.
        It assumes that you are feeding an array of floats, ints, etc.

    Args:
        data (:obj:`numpy.array`): Array to be evaluated

    Return:
        isherm (:obj:`bool`): Is the array hermitian
    '''
    if isinstance(data, (list, tuple)): data = np.array(data)
    if np.iscomplex(data).any():
        raise TypeError("The data type that was detected in the passed data were " \
                        +"complex instead or real values. Use isantihermitian instead.")
    a = -1*(np.ones(data.shape) - np.eye(data.shape[0])) + np.eye(data.shape[0])
    antisymm = a*np.transpose(data)
    isantisymm = np.allclose(antisymm, data)
    return isantisymm

@vectorize([float64(complex128)])
def abs2(x):
    '''Get the square of a complex number'''
    return np.square(np.real(x)) + np.square(np.imag(x))

