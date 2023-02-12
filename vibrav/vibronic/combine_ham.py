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
from vibrav.util.io import open_txt
import pandas as pd
import numpy as np
import warnings
import os

def combine_ham_files(paths, nmodes, out_path='confg{:03d}', debug=False):
    '''
    Helper script to combine the Hamiltonian files of several claculations.
    This is helpful as one can calculate the Hamiltonian elements of a
    displaced structure separately for each spin as the Hamiltonian elements
    between different spin-multiplicities are 0 by definition and this can
    drastically reduce the computational complexity. This function can grab
    the different `'ham-sf.txt'` files in each of the specified paths and
    combine them into one gigantic `'ham-sf.txt'` file. The resulting
    `'ham-sf.txt'` files will be written to the given path with the same
    indexing scheme.

    Note:
        The ordering of the multiplicities will be inferred from the ordering
        of the paths. That is to say, whichever order the paths are given will
        be the order of the multiplicities.

    Args:
        paths (:obj:`list`): List of the paths to where the `'ham-sf.txt'` files
                             are located for each of the multiplicites to combine.
                             **Must be able to use the python format function for
                             these and the proper padding for the indexing must be
                             used. NOTHING is assumed.**
        nmodes (:obj:`int`): Number of normal modes in the molecule.
        out_path (:obj:`str`, optional): String object to folders where the new
                                         `'ham-sf.txt'` files will be written to.
                                         **Must be in the format to use the
                                         python `format` function**. Defaults to
                                         `'confg{:03d}'`.
        debug (:obj:`bool`, optional): Turn on some light debug text.
    '''
    class FileNotFound(Exception):
        pass
    # loop over all of the displaced structures that there should exist
    for idx in range(2*nmodes + 1):
        try:
            files = []
            # loop over the given paths one by one to build the array of data
            for path in paths:
                # check if the given path is contains the file name or they are
                # just the directory names
                if not path.endswith('ham-sf.txt'):
                    dirname = path.format(idx)
                else:
                    dirname = os.path.join(*path.split(os.sep)[:-1])
                    dirname = dirname.format(idx)
                # check that the directory exists
                if not os.path.exists(dirname):
                    text = "Directory {} not found. Skipping index...."
                    warnings.warn(text.format(dirname), Warning)
                    raise FileNotFound
                # check that the ham-sf.txt file exists
                file = os.path.join(dirname, 'ham-sf.txt')
                if not os.path.exists(file):
                    text = "Missing 'ham-sf.txt' file in path {} for index {}. Skipping index...."
                    warnings.warn(text.format(dirname, idx), Warning)
                    raise FileNotFound
                files.append(file)
        except FileNotFound:
            continue
        # read and combine the files
        df = block_diagonal(files)
        # write the file
        if not os.path.exists(out_path.format(idx)):
            os.mkdir(out_path.format(idx))
        filename = os.path.join(out_path.format(idx), 'ham-sf.txt')
        if debug:
            text = "Writting 'ham-sf.txt' file to {}".format(filename)
            print(text)
        write_txt(filename, non_matrix=True)

