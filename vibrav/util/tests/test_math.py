import numpy as np
import pytest
from vibrav.util.math import ishermitian, isantihermitian, issymmetric, isantisymmetric

@pytest.mark.parametrize('arr', [([[1, -4], [-4, -3]]), ([[1, 7+3j], [7-3j, 7]])])
def test_ishermitian(arr):
    assert ishermitian(arr)
 
@pytest.mark.parametrize('arr', [([[1, -4], [4, -3]]), ([[1, 7+3j], [-7+3j, 7]])])
def test_isantihermitian(arr):
    assert isantihermitian(arr)

@pytest.mark.parametrize('arr', [([[1, -4], [-4, -3]]), ([[1, 7+3j], [7+3j, 7]])])
def test_issymmetric(arr):
    assert issymmetric(arr)

@pytest.mark.parametrize('arr', [([[1, -4], [4, -3]]), ([[1, 7+3j], [-7+3j, 7]])])
def test_isantisymmetric(arr):
    assert isantisymmetric(arr)

