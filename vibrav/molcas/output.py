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
from exa.core import Editor
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
            for ndx, (start, stop) in enumerate(zip(starts[:n], stops[:n])):
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
        props = np.array(found)[:3]
        stop = props[0] + 5
        while self[stop].strip(): stop += 1
        data_length = stop - props[0] - 5
        # get the data
        stdm = self._property_parsing(props, data_length)
        stdm['component'] = np.repeat(['x', 'y', 'z'], data_length)
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
        found = self.find(_reenerg)
        if not found:
            return
        energies = []
        for _, line in found:
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
            return
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

