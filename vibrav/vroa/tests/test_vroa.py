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
import tarfile
import os
import shutil


def test_vroa():
    with tarfile.open(resource('nwchem-h2o2-vroa.tar.gz'), 'r:gz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)
    parent = os.getcwd()
    os.chdir('nwchem-h2o2-vroa')
    cls = VROA(config_file='va.conf')
    cls.vroa(atomic_units=True)
    base_scatter = pd.read_csv('final-scatter.csv', index_col=False)
    cols = ['backscatter', 'forwardscatter']
    test = cls.scatter.copy()
    assert np.allclose(base_scatter[cols[0]], test[cols[0]])
    assert np.allclose(base_scatter[cols[1]], test[cols[1]])
    os.chdir(parent)
    shutil.rmtree('nwchem-h2o2-vroa')

