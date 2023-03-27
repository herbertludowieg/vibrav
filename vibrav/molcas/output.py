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
from exatomic.exa.core.editor import Editor
import pandas as pd
import numpy as np

class Output(Editor):
    '''
    This output editor is supposed to work for OpenMolcas.
    Currently it is only designed to parse the data required for this script.
    '''
    _resta = "STATE"
    def _property_parsing(self, props, data_length):
        ''' Helper method for parsing the spin-free properties sections. '''
        all_dfs = []
        # this is a bit of a mess but since we have three components
        # we can take a nested for loop of three elements without too
        # much of a hit on performance
        for idx, prop in enumerate(props):
            # keep track of columns parsed so far
            counter = 0
            # find where the data blocks are printed
            starts = np.array(self.find(self._resta, start=prop, keys_only=True)) + prop + 2
            # data_length should always be the same
            # we could determine it for each data block but that would
            # be a large number of while loops
            stops = starts + data_length
            # hardcoding but should apply in all of our cases
            # there should be a max of 4 columns of data in each of the 'STATE'
            # data blocks so we get the maximum number of hits that there should be
            # assuming a square matrix of data_length x data_length
            n = int(np.ceil(data_length/4))
            dfs = []
            # grab all of the data
            for start, stop in zip(starts[:n], stops[:n]):
                ncols = len(self[start-2].split())
                df = self.pandas_dataframe(start, stop, ncol=ncols)
                df[0] -= 1
                # set the indexes as they may be different and drop the column
                df.index = df[0]
                df.drop(0, axis=1, inplace=True)
                # set the columns as they should be
                df.columns = list(range(counter, counter+ncols-1))
                dfs.append(df)
                counter += ncols - 1
            # put the component together
            all_dfs.append(pd.concat(dfs, axis=1))
            all_dfs[-1]['component'] = idx
        df = pd.concat(all_dfs, ignore_index=True)
        return df

    def _oscillator_parsing(self, start_idx):
        ''' Helper method to parse the oscillators. '''
        ldx = start_idx + 6
        oscillators = []
        while '-----' not in self[ldx]:
            oscillators.append(self[ldx].split())
            ldx += 1
        df = pd.DataFrame(oscillators)
        df.columns = ['nrow', 'ncol', 'oscil', 'a_x', 'a_y', 'a_z', 'a_tot']
        df[['nrow', 'ncol']] = df[['nrow', 'ncol']].astype(np.uint16)
        df[['nrow', 'ncol']] -= [1, 1]
        df[['nrow', 'ncol']] = df[['nrow', 'ncol']].astype('category')
        cols = ['oscil', 'a_x', 'a_y', 'a_z', 'a_tot']
        df[cols] = df[cols].astype(np.float64)
        return df

    def parse_sf_dipole_moment(self):
        '''
        Get the Spin-Free electric dipole moment.

        Raises:
            AttributeError: If it cannot find the angular momentum property. This is
                            applicable to this package as it expects it to be present.
        '''
        # define the search string
        _retdm = "PROPERTY: MLTPL  1"
        _resta = "STATE"
        component_map = {0: 'x', 1: 'y', 2: 'z'}
        found = self.find(_retdm, keys_only=True)
        if not found:
            raise AttributeError("Could not find the TDM in the output")
        #found = self.find(_retdm, _resta, keys_only=True)
        if len(found) > 6:
            props = np.array(found)[:6:2]
        else:
            props = np.array(found)[:3]
        stop = props[0] + 5
        while self[stop].strip(): stop += 1
        data_length = stop - props[0] - 5
        # get the data
        stdm = self._property_parsing(props, data_length)
        stdm['component'] = stdm['component'].map(component_map)
        self.sf_dipole_moment = stdm

    def parse_sf_quadrupole_moment(self):
        '''
        Get the Spin-Free electric quadrupole moment.

        Raises:
            AttributeError: If it cannot find the angular momentum property. This is
                            applicable to this package as it expects it to be present.
        '''
        _requad = "PROPERTY: MLTPL  2"
        _resta = "STATE"
        component_map = {0: 'xx', 1: 'xy', 2: 'xz', 3: 'yy', 4: 'yz', 5: 'zz'}
        found = self.find(_requad, keys_only=True)
        if not found:
            raise AttributeError("Could not find the Quadrupoles in the output")
        props = np.array(found)[:6]
        stop = props[0] + 5
        while self[stop].strip(): stop += 1
        data_length = stop - props[0] - 5
        # get the data
        sqdm = self._property_parsing(props, data_length)
        sqdm['component'] = sqdm['component'].map(component_map)
        self.sf_quadrupole_moment = sqdm

    def parse_sf_angmom(self):
        '''
        Get the Spin-Free angular momentum.

        Raises:
            AttributeError: If it cannot find the angular momentum property. This is
                            applicable to this package as it expects it to be present.
        '''
        _reangm = "PROPERTY: ANGMOM"
        _resta = "STATE"
        component_map = {0: 'x', 1: 'y', 2: 'z'}
        found = self.find(_reangm, keys_only=True)
        if not found:
            raise AttributeError("Could not find the Angular Momentum in the output")
        props = np.array(found)[:3]
        stop = props[0] + 5
        while self[stop].strip(): stop += 1
        data_length = stop - props[0] - 5
        # get the data
        sangm = self._property_parsing(props, data_length)
        sangm['component'] = sangm['component'].map(component_map)
        self.sf_angmom = sangm

    def parse_sf_energy(self):
        '''
        Get the Spin-Free energies.
        '''
        _reenerg = " RASSI State "
        _reenerg_rasscf = " RASSCF root number"
        found = self.find(_reenerg, _reenerg_rasscf)
        key = ''
        if found[_reenerg]:
            key = _reenerg
        elif found[_reenerg_rasscf]:
            key = _reenerg_rasscf
        else:
            return
        if key == '':
            raise ValueError("This should not have executed at all")
        energies = []
        for _, line in found[key]:
            energy = float(line.split()[-1])
            energies.append(energy)
        rel_energy = list(map(lambda x: x - energies[0], energies))
        df = pd.DataFrame.from_dict({'energy': energies, 'rel_energy': rel_energy})
        self.sf_energy = df

    def parse_so_energy(self):
        '''
        Get the Spin-Orbit energies.
        '''
        _reenerg = " SO-RASSI State "
        found = self.find(_reenerg)
        if not found:
            raise AttributeError("Could not find the Spin-Orbit energies.")
        energies = []
        for _, line in found:
            energy = float(line.split()[-1])
            energies.append(energy)
        rel_energy = list(map(lambda x: x - energies[0], energies))
        df = pd.DataFrame.from_dict({'energy': energies, 'rel_energy': rel_energy})
        self.so_energy = df

    def parse_sf_oscillator(self):
        '''
        Get the printed Spin-Free oscillators.
        '''
        _reosc = "++ Dipole transition strengths (spin-free states):"
        found = self.find(_reosc, keys_only=True)
        if not found:
            return
        if len(found) > 1:
            raise NotImplementedError("We have found more than one key for the spin-free " \
                                      +"oscillators.")
        df = self._oscillator_parsing(found[0])
        self.sf_oscillator = df

    def parse_so_oscillator(self):
        '''
        Get the printed Spin-Orbit oscillators.
        '''
        _reosc = "++ Dipole transition strengths (SO states):"
        found = self.find(_reosc, keys_only=True)
        if not found:
            return
        if len(found) > 1:
            raise NotImplementedError("We have found more than one key for the spin-orbit " \
                                      +"oscillators.")
        df = self._oscillator_parsing(found[0])
        self.so_oscillator = df

    def parse_contribution(self):
        '''
        Parse the Spin-Free contibutions to each Spin-Orbit state from a regular molcas
        Spin-Orbit RASSI calculation.
        '''
        _recont = "Weights of the five most important spin-orbit-free states for each spin-orbit state."
        found = self.find(_recont, keys_only=True)
        if not found:
            return
        if len(found) > 1:
            raise NotImplementedError("Who do I look like Edward Snowden?")
        start = found[0] + 4
        end = found[0] + 4
        while '-----' not in self[end]: end += 1
        df = self.pandas_dataframe(start, end, ncol=17)
        so_state = df[0].values
        energy = df[1].values
        df = pd.DataFrame(df[range(2,17)].values.reshape(df.shape[0]*5, 3))
        df.columns = ['sf_state', 'spin', 'weight']
        so_state = np.repeat(so_state, 5)
        energy = np.repeat(energy, 5)
        df['so_state'] = pd.Categorical(so_state)
        df['energy'] = energy.astype(np.double)
        df['sf_state'] = pd.Categorical(df['sf_state'].astype(int))
        df['spin'] = df['spin'].astype(np.half)
        df['weight'] = df['weight'].astype(np.single)
        self.contribution = df

    def parse_rasscf_hamiltonian(self, last_frame=True):
        # parsing keys
        _reham = "Explicit Hamiltonian"
        found = self.find(_reham, keys_only=True)
        # we assume that you know that these should be in the output
        if not found:
            text = "Could not find the Hamiltonian in the RASSCF output. " \
                   +"Make sure that you used print level 5."
            raise ValueError(text)
        # get the size of the matrix
        matrix_size = list(map(int, self[found[0]+1].replace('x', '').split()[-2:]))
        if matrix_size[0] != matrix_size[1]:
            raise NotImplementedError("Sorry there is only support for square matrices")
        # get where the matrix starts
        starts = np.array(found)+3
        end = starts[0]
        while self[end].strip(): end += 1
        ends = starts + (end - starts[0])
        dfs = []
        if last_frame:
            df = self.pandas_dataframe(starts[-1], ends[-1], ncol=5)
            df = df.values.reshape(-1)
            df = df[~np.isnan(df)]
            full = matrix_size[0]*matrix_size[1]
            lowtri = matrix_size[0]*(matrix_size[1]+1)/2
            # determine if we have a square or lower triangular matrix
            if df.shape[0] == full:
                lower_triangular = False
            elif df.shape[0] != lowtri and df.shape[0] != full:
                text = "Parsed Hamiltonian matrix does not have the right number of elements. " \
                       +"Expected {}, currently {}"
                lower_triangular = False
                raise ValueError(text.format(lowtri, df.shape[0]))
            else:
                lower_triangular = True
            # deal with the matrix correctly
            if lower_triangular:
                il = np.tril_indices(matrix_size[0])
                tmp = np.zeros(matrix_size)
                tmp[il] = df
                tmp2 = tmp + tmp.T - (np.eye(matrix_size[0])*tmp)
                df = pd.DataFrame(tmp2)
            else:
                df = pd.DataFrame(df.reshape(matrix_size))
            self.hamiltonian = df
        else:
            for idx, (start, end) in enumerate(zip(starts, ends)):
                df = self.pandas_dataframe(start, end, ncol=5)
                df = df.values.reshape(-1)
                df = df[~np.isnan(df)]
                full = matrix_size[0]*matrix_size[1]
                lowtri = matrix_size[0]*(matrix_size[1]+1)/2
                # determine if we have a square or lower triangular matrix
                if df.shape[0] == full:
                    lower_triangular = False
                elif df.shape[0] != lowtri and df.shape[0] != full:
                    text = "Parsed Hamiltonian matrix does not have the right number of elements. " \
                           +"Expected {}, currently {}"
                    lower_triangular = False
                    raise ValueError(text.format(lowtri, df.shape[0]))
                else:
                    lower_triangular = True
                # deal with the matrix correctly
                if lower_triangular:
                    il = np.tril_indices(matrix_size[0])
                    tmp = np.zeros(matrix_size)
                    tmp[il] = df
                    tmp2 = tmp + tmp.T - (np.eye(matrix_size[0])*tmp)
                    df = pd.DataFrame(tmp2)
                else:
                    df = pd.DataFrame(df.reshape(matrix_size))
                # add a frame column to keep track of stuff
                df['frame'] = idx
                dfs.append(df)
            # put it all together
            ham = pd.concat(dfs, ignore_index=True)
            self.hamiltonian = ham

    def parse_rasscf_eigenvalues(self, last_frame=True):
        # parsing keys
        _reeigval = "Eigenvalues of the explicit Hamiltonian"
        found = self.find(_reeigval, keys_only=True)
        # we assume that you know that these should be in the output
        if not found:
            text = "Could not find the Eigenvalues in the RASSCF output. " \
                   +"Make sure that you used print level 5."
            raise ValueError(text)
        # get the number of eigenvalues there should be
        size = int(self[found[0]+2].split()[-1])
        starts = np.array(found)+4
        end = starts[0]
        while self[end].strip(): end += 1
        ends = starts + (end - starts[0])
        dfs = []
        if last_frame:
            arrs = []
            # read through every line and get the eigenvalues correctly
            for ldx in range(starts[-1], ends[-1]):
                line = self[ldx]
                d = line.split('-')
                vals = list(map(lambda x: -1*float(x), d[1:]))
                arrs.append(vals)
            df = pd.Series(np.concatenate(arrs))
            if df.shape[0] != size:
                text = "Parsed Eigenvalues do not have the right number of elements. " \
                       +"Expected {}, currently {}"
                raise ValueError(text.format(size, df.shape[0]))
            self.eigenvalues = df
        else:
            for idx, (start, end) in enumerate(zip(starts, ends)):
                arrs = []
                # read through every line and get the eigenvalues correctly
                for ldx in range(start, end):
                    line = self[ldx]
                    d = line.split('-')
                    vals = list(map(lambda x: -1*float(x), d[1:]))
                    arrs.append(vals)
                df = pd.DataFrame.from_dict({'values': np.concatenate(arrs), 'frame': idx})
                if df.shape[0] != size:
                    text = "Parsed Eigenvalues do not have the right number of elements. " \
                           +"Expected {}, currently {}"
                    raise ValueError(text.format(size, df.shape[0]))
                # add a frame column to keep track of stuff
                df['frame'] = idx
                dfs.append(df)
            # put it all together
            values = pd.concat(dfs, ignore_index=True)
            self.eigenvalues = values

    def parse_rasscf_eigenvectors(self, last_frame=True):
        # parsing keys
        _reeigvec = "Eigenvectors of the explicit Hamiltonian"
        found = self.find(_reeigvec, keys_only=True)
        # we assume that you know that these should be in the output
        if not found:
            text = "Could not find the Eigenvectors in the RASSCF output. " \
                   +"Make sure that you used print level 5."
            raise ValueError(text)
        # ensure square matrix
        matrix_size = list(map(int, self[found[0]+1].replace('x', '').split()[-2:]))
        if matrix_size[0] != matrix_size[1]:
            raise NotImplementedError("Sorry there is only support for square matrices")
        starts = np.array(found)+2
        end = starts[0]
        while "Initial" not in self[end].strip(): end += 1
        ends = starts + (end - starts[0])
        size = matrix_size[0]*matrix_size[1]
        dfs = []
        if last_frame:
            # get data
            vec = self.pandas_dataframe(starts[-1], ends[-1], ncol=5).values.reshape(-1)
            vec = vec[~np.isnan(vec)]
            # ensure size
            if vec.shape[0] != size:
                text = "Parsed Eigenvector matrix does not have the right number of elements. " \
                       +"Expected {}, currently {}"
                raise ValueError(text.format(size, vec.shape[0]))
            df = pd.DataFrame(vec.reshape(matrix_size))
            self.eigenvectors = df
        else:
            for idx, (start, end) in enumerate(zip(starts, ends)):
                # get data
                vec = self.pandas_dataframe(start, end, ncol=5).values.reshape(-1)
                vec = vec[~np.isnan(vec)]
                # ensure size
                if vec.shape[0] != size:
                    text = "Parsed Eigenvector matrix does not have the right number of elements. " \
                           +"Expected {}, currently {}"
                    raise ValueError(text.format(size, vec.shape[0]))
                df = pd.DataFrame(vec.reshape(matrix_size))
                df['frame'] = idx
                dfs.append(df)
            # put it all together
            self.eigenvectors = pd.concat(dfs, ignore_index=True)

    def parse_rasscf_ordering(self):
        _reorder = "Configurations included in the explicit Hamiltonian"
        found = self.find(_reorder, keys_only=True)
        if not found:
            text = "Could not find the energy ordering in the RASSCF output. " \
                   +"Make sure that you used print level 5."
            raise ValueError(text)
        #size = int(self[found[0]+2].split()[-1])
        start = found[0]+4
        end = start
        while self[end].strip(): end += 1
        order = self.pandas_dataframe(start, end, ncol=20).values.reshape(-1)
        order = order[~np.isnan(order)]
        self.order = order.astype(int)-1

    def parse_rasscf(self, last_frame=True):
        if not hasattr(self, 'hamiltonian'):
            self.parse_rasscf_hamiltonian(last_frame)
        if not hasattr(self, 'eigenvectors'):
            self.parse_rasscf_eigenvectors(last_frame)
        if not hasattr(self, 'eigenvalues'):
            self.parse_rasscf_eigenvalues(last_frame)

