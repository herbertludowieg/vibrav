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

