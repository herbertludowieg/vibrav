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
import os
import vibrav
import numpy as np

def get_static_path():
    return os.sep.join(vibrav.__file__.split(os.sep)[:-1]+['static'])

def resource(file):
    abspath, files = list_resource(full_path=True, return_both=True)
    index = []
    for idx, f in enumerate(files):
        if f == file:
            index.append(idx)
    if len(index) > 1:
        raise ValueError("More than one file was found with that name in the static directory. " \
                        +"Please submit an issue on the github page. If this was a file that you " \
                        +"added make sure that it does not have the same name as some of the " \
                        +"existing files, regardless of whether they are in different directories.")
    elif len(index) == 0:
        raise FileNotFoundError("The specified resource file was not found")
    return abspath[index[0]]

def list_resource(full_path=False, return_both=False, search_string=''):
    fp = []
    if full_path:
        abspaths = []
    else:
        abspaths = None
    for (dirpath, _, files) in os.walk(get_static_path()):
        paths = list(map(lambda x: os.path.abspath(os.path.join(dirpath, x)), files))
        for path, file in zip(paths, files):
            if os.path.isfile(path) and search_string in path:
                if not abspaths is None:
                    abspaths.append(path)
                fp.append(file)
    if full_path and not return_both:
        return abspaths
    elif not (full_path and return_both):
        return fp
    elif return_both:
        return abspaths, fp
    else:
        raise RuntimeError("Um.....this is embarrasing. " \
                          +"The input for list_resource was not understood")

