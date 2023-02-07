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
from vibrav.zpvc import ZPVC
from vibrav.base import resource
import numpy as np
import pandas as pd
import pytest
import tarfile
import os
import glob

@pytest.fixture
def zpvc_results():
    df = pd.read_csv(resource('nitromal-zpvc-results.csv.xz'), compression='xz',
                     index_col=0, header=0)
    zpvc_results = df
    yield zpvc_results

@pytest.fixture
def zpvc_geometry():
    df = pd.read_csv(resource('nitromal-zpvc-geometry.csv.xz'), compression='xz',
                     index_col=False)
    zpvc_geometry = df.groupby('temp')
    yield zpvc_geometry

def test_zpvc(zpvc_results, zpvc_geometry):
    zpvc = ZPVC(config_file=resource('nitromal-zpvc-va.conf'))
    zpvc.zpvc(write_out_files=False)
    cols = ['tot_anharm', 'tot_curva', 'zpvc', 'zpva']
    assert np.allclose(zpvc_results[cols].values, zpvc.zpvc_results[cols].values)
    cols = ['x', 'y', 'z']
    for t in zpvc.config.temperature:
        assert np.allclose(zpvc_geometry.get_group(t)[cols],
                           zpvc.eff_coord.groupby('temp').get_group(t)[cols])

