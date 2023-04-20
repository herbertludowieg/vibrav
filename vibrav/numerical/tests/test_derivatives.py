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
from vibrav.base import resource
from vibrav.numerical.derivatives import (two_point_1d, four_point_1d,
                                          six_point_1d, _get_arb_coeffs,
                                          _determine_prefactors,
                                          arb_disps_1d)
import pandas as pd
import numpy as np
import pytest

@pytest.mark.parametrize('steps,actual',([np.array([1]), [0.5]],
                                         [np.array([1,2]), [2./3, -1./12]],
                                         [np.array([1,2,3]), [3./4, -3./20, 1./60]],
                                         [np.array([1,2,3,4]), [4./5, -1./5, 4./105, -1./280]]))
def test_get_arb_coeffs(steps, actual):
    coeffs = _get_arb_coeffs(steps)
    assert np.allclose(coeffs, actual)

