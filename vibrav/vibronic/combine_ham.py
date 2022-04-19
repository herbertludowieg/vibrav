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
    from vibrav.util.io import open_txt
    import pandas as pd
    import warnings
    import os
    class FileNotFound(Exception):
        pass
    # loop over all of the displaced structures that there should exist
    for idx in range(2*nmodes + 1):
        size_count = 0
        dfs = []
        try:
            # loop over the given paths one by one to build the array of data
            for path in paths:
                # check if the given path is contains the file name or they are
                # just the directory names
                if not path.endswith('ham-sf.txt'):
                    dir = path.format(idx)
                else:
                    dir = os.path.join(*path.split(os.sep)[:-1])
                    dir = dir.format(idx)
                # check that the directory exists
                if not os.path.exists(dir):
                    text = "Directory {} not found. Skipping index...."
                    warnings.warn(text.format(dir), Warning)
                    raise FileNotFound
                # check that the ham-sf.txt file exists
                file = os.path.join(dir, 'ham-sf.txt')
                if not os.path.exists(file):
                    text = "Missing 'ham-sf.txt' file in path {} for index {}. Skipping index...."
                    warnings.warn(text.format(dir, idx), Warning)
                    raise FileNotFound
                # read the data
                data = open_txt(file, rearrange=False)
                # ensure that the data has the right number of columns
                # can be a possibility if Molcas decides to change something with
                # writing the ham-sf.txt output files
                if data.columns.shape[0] != 4:
                    text = "Did not find exactly four column labels in file {}"
                    raise ValueError(text.format(file))
                if not all(data.columns == ['nrow', 'ncol', 'real', 'imag']):
                    text = "Found an inconsistency in the column labels for file {}"
                    if debug:
                        print(data.columns)
                        print(data.columns == ['nrow', 'ncol', 'real', 'imag'])
                    raise ValueError(text.format(file))
                size_count += data.shape[0]
                if not dfs:
                    # append if there is nothing in the dfs array we initialize it
                    # with the very first entry
                    dfs.append(data)
                else:
                    # otherwise change the values of nrow and ncol to be placed on the
                    # hamiltonian matrix correctly
                    # we get the max values from the last array as that is the starting
                    # point for the next one
                    max_row = dfs[-1]['nrow'].max() + 1
                    max_col = dfs[-1]['ncol'].max() + 1
                    nrow = data['nrow'].unique().shape[0]
                    ncol = data['ncol'].unique().shape[0]
                    data['nrow'] = np.tile(range(max_row, max_row + nrow), ncol)
                    data['ncol'] = np.repeat(range(max_col, max_col + ncol), nrow)
                    dfs.append(data)
            # put it all together
            df = pd.concat(dfs, ignore_index=True)
            df['nrow'] += 1
            df['ncol'] += 1
            # ensure that we have the right size
            if df.shape[0] != size_count:
                text = "The final size of the Hamiltonian matrix does not match the sum of " \
                       +"the parts. Currently {}, expected {}"
                raise ValueError(text.format(df.shape[0], size_count))
            # write the data to file
            if not os.path.exists(out_path.format(idx)):
                os.mkdir(out_path.format(idx))
            filename = os.path.join(out_path.format(idx), 'ham-sf.txt')
            if debug:
                text = "Writting 'ham-sf.txt' file to {}".format(filename)
                print(text)
            head_temp = '{:<6s}  {:<6s}  {:>23s}  {:>23s}\n'
            data_temp = '{:>6d}  {:>6d}  {:>23.16E}  {:>23.16E}\n'
            with open(filename, 'w') as fn:
                fn.write(head_temp.format('#NROW', 'NCOL', 'REAL', 'IMAG'))
                for row, col, real, imag in zip(df.nrow, df.ncol, df.real, df.imag):
                    fn.write(data_temp.format(row, col, real, imag))
        except FileNotFound:
            continue

