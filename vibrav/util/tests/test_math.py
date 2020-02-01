import numpy as np
import pytest
from vibrav.util.math import ishermitian

@pytest.mark.parametrize('arr', [([[1, -4], [-4, -3]]), ([[1, 7+3j], [7-3j, 7]])])
def test_ishermitian(arr):
    assert ishermitian(arr)
 
