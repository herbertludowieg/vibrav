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
from vibrav.vroa import VROA
from vibrav.base import resource
import numpy as np
import pandas as pd

def test_vroa():
    vroa = VROA(config_file=resource('h2o2-vroa-va.conf'))
    vroa.vroa(atomic_units=True)
    base_scatter = pd.read_csv(resource('h2o2-vroa-final-scatter.csv.xz'),
                               index_col=False, compression='xz')
    cols = ['backscatter', 'forwardscatter']
    test = vroa.scatter.copy()
    assert np.allclose(base_scatter[cols[0]], test[cols[0]])
    assert np.allclose(base_scatter[cols[1]], test[cols[1]])

