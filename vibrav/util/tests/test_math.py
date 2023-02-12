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
from vibrav.util.math import (ishermitian, isantihermitian, issymmetric,
                              isantisymmetric, get_triu, get_tril, abs2)
import numpy as np
import pytest

@pytest.mark.parametrize('arr', [([[1, -4], [-4, -3]]),
                                 ([[1, 7+3j], [7-3j, 7]])])
def test_ishermitian(arr):
    assert ishermitian(arr)
 
@pytest.mark.parametrize('arr', [([[1, -4], [4, -3]]),
                                 ([[1, 7+3j], [-7+3j, 7]])])
def test_isantihermitian(arr):
    assert isantihermitian(arr)

@pytest.mark.parametrize('arr', [([[1, -4], [-4, -3]]),
                                 ([[1, 7+3j], [7+3j, 7]])])
def test_issymmetric(arr):
    if not np.iscomplex(arr).any():
        assert issymmetric(arr)
    else:
        with pytest.raises(TypeError):
            assert issymmetric(arr)

@pytest.mark.parametrize('arr', [([[1, -4], [4, -3]]),
                                 ([[1, 7+3j], [-7+3j, 7]])])
def test_isantisymmetric(arr):
    if not np.iscomplex(arr).any():
        assert isantisymmetric(arr)
    else:
        with pytest.raises(TypeError):
            assert isantisymmetric(arr)

@pytest.mark.parametrize('arr,k,actual', [([[1,2], [3,4]], 0, [1,2,4]),
                                   ([[1,2,3],[4,5,6],[7,8,9]], 0, [1,2,3,5,6,9]),
                                   ([[1,2,3],[4,5,6],[7,8,9]], 1, [2,3,6]),
                                   ([[1,2,3],[4,5,6],[7,8,9]], -1, [1,2,3,4,5,6,8,9]),
                                   ([[1,2,3],[4,5,6]], 0, [1,2,3,5,6])])
def test_get_triu(arr, k, actual):
    triu_arr = get_triu(arr, k)
    assert np.allclose(triu_arr, actual)

@pytest.mark.parametrize('arr,k,actual', [([[1,2], [3,4]], 0, [1,3,4]),
                                   ([[1,2,3],[4,5,6],[7,8,9]], 0, [1,4,5,7,8,9]),
                                   ([[1,2,3],[4,5,6],[7,8,9]], 1, [1,2,4,5,6,7,8,9]),
                                   ([[1,2,3],[4,5,6],[7,8,9]], -1, [4,7,8]),
                                   ([[1,2,3],[4,5,6]], 0, [1,4,5])])
def test_get_tril(arr, k, actual):
    tril_arr = get_tril(arr, k)
    assert np.allclose(tril_arr, actual)

@pytest.mark.parametrize('val,actual', [(1+2j, 5), (2j, 4), (4, 16),
                                        (4j+2, 20)])
def test_abs2(val, actual):
    test = abs2(val)
    assert np.allclose(test, actual)

