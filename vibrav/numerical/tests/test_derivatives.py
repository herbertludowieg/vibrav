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
from vibrav.numerical.derivatives import (two_point_1d, four_point_1d, six_point_1d, eight_point_1d,
                                          two_point_2d, four_point_2d, six_point_2d, eight_point_2d,
                                          _get_arb_coeffs)
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

dx_map = dict(two=two_point_1d, four=four_point_1d, six=six_point_1d,
              eight=eight_point_1d)
dx2_map = dict(two=two_point_2d, four=four_point_2d, six=six_point_2d,
               eight=eight_point_2d)
idx_map = dict(two=1, four=2, six=3, eight=4)

@pytest.mark.parametrize('method,delta,index',
                         (['two', 0.1, 0], ['four', 0.1, 0], ['six', 0.1, 0], ['eight', 0.1, 0],
                          ['two', 0.5, 1], ['four', 0.5, 1], ['six', 0.5, 1], ['eight', 0.5, 1],
                          ['two', 1.0, 2], ['four', 1.0, 2], ['six', 1.0, 2], ['eight', 1.0, 2]))
def test_1d_derivs(method, delta, index):
    base = 6.5
    steps = 10
    x = np.linspace(base-delta*steps/2,base+delta*steps/2,steps+1)
    y = np.log(x)
    y_plus = y[-int(steps/2):]
    y_minus = np.flip(y[:int(steps/2)])
    deriv = dx_map[method](y_plus[:idx_map[method]], y_minus[:idx_map[method]], delta)
    df = pd.read_csv(resource('lnx-1d-derivs.csv'), index_col=0, header=0)
    test = df.groupby('index').get_group(index)[method+'-point'].values[0]
    assert abs(deriv - test) < 1e-6

@pytest.mark.parametrize('method,delta,index',
                         (['two', 0.1, 0], ['four', 0.1, 0], ['six', 0.1, 0], ['eight', 0.1, 0],
                          ['two', 0.5, 1], ['four', 0.5, 1], ['six', 0.5, 1], ['eight', 0.5, 1],
                          ['two', 1.0, 2], ['four', 1.0, 2], ['six', 1.0, 2], ['eight', 1.0, 2]))
def test_2d_derivs(method, delta, index):
    base = 6.5
    steps = 10
    x = np.linspace(base-delta*steps/2,base+delta*steps/2,steps+1)
    y = np.log(x)
    y_plus = y[-int(steps/2):]
    y_minus = np.flip(y[:int(steps/2)])
    y_equil = y[int(steps/2)]
    deriv = dx2_map[method](y_plus[:idx_map[method]], y_minus[:idx_map[method]],
                            y_equil, delta)
    df = pd.read_csv(resource('lnx-2d-derivs.csv'), index_col=0, header=0)
    test = df.groupby('index').get_group(index)[method+'-point'].values[0]
    assert abs(deriv - test) < 1e-6

