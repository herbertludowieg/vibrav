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
from vibrav.numerical.degeneracy import energetic_degeneracy
from vibrav.base import resource
import numpy as np
import pandas as pd

def test_energetic_degeneracy():
    degen = pd.read_csv(resource('molcas-rassi-nien-degen-so-energy.csv.xz'), compression='xz',
                        index_col=False)
    df = pd.read_csv(resource('molcas-rassi-nien-energy.csv.xz'), compression='xz', index_col=False)
    test_degen = energetic_degeneracy(df['so'].values, 1e-5)
    cols = ['value', 'degen']
    assert np.allclose(degen[cols].values, test_degen[cols].values)

