# -*- coding: utf-8 -*-
# Copyright 2019-2020 Herbert D. Ludowieg
# Distributed under the terms of the Apache License 2.0
import numpy as np

def get_triu(arr, k=0):
    '''
    Get the upper triangular indeces of the input matrix

    Args:
        arr (np.array or list-like object): Array to parse the upper triangular elements
        k (int): k parameter that goes into the `np.triu_indices_from` function. Refer to numpy
                 documentation for more information.

    Returns:
        triu_arr (np.array): 1D array with the upper triangular elements
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
        data (np.array or list-like object): Array to be evaluated

    Return:
        isherm (bool): Is the array hermitian
    '''
    herm = np.conjugate(np.transpose(data))
    isherm = np.allclose(herm, data)
    return isherm

