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
'''
Base module
###########
Handles the resource files.
'''
import os
import vibrav
import numpy as np

def _get_static_path():
    return os.sep.join(vibrav.__file__.split(os.sep)[:-1]+['static'])

def resource(file):
    '''
    Get the requested resource file from the static directory.

    Args:
        file (:obj:`str`): Name of resource file.

    Returns:
        resource_path (:obj:`str`): Absolute path to the resource file.

    Raises:
        ValueError: When there is more than one resource file found with that same name.
        FileNotFoundError: When the resource file cannot be found in the static directory.
    '''
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
    resource_path = abspath[index[0]]
    return resource_path

def list_resource(full_path=False, return_both=False, search_string=''):
    '''
    Get all of the available resource files in the static directory.

    Args:
        full_path (:obj:`bool`, optional): Return the absolute path of the resource files.
                                           Defaults to :code:`False`.
        return_both (:obj:`bool`, optional): Return both the absolute paths and resource files.
                                             Defaults to :code:`False`.
        search_string (:obj:`str`, optional): Regex string to limit the number of entries to return.
                                              Defaults to :code:`''`.

    Returns:
        resource_files (:obj:`list`): Resource file list depending on the input parameters.
    '''
    fp = []
    if full_path:
        abspaths = []
    else:
        abspaths = None
    for (dirpath, _, files) in os.walk(_get_static_path()):
        paths = list(map(lambda x: os.path.abspath(os.path.join(dirpath, x)), files))
        for path, file in zip(paths, files):
            if os.path.isfile(path) and search_string in path:
                if not abspaths is None:
                    abspaths.append(path)
                fp.append(file)
    if return_both:
        resource_files = [abspaths, fp]
    elif full_path and not return_both:
        resource_files = abspaths
    elif not (full_path and return_both):
        resource_files = fp
    else:
        raise RuntimeError("Um.....this is embarrasing. " \
                          +"The input for list_resource was not understood")
    return resource_files

