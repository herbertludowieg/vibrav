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
import pytest
from vibrav.util.math import ishermitian, isantihermitian, issymmetric, isantisymmetric

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

