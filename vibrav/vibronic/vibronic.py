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
import pandas as pd
import numpy as np
import os
import warnings
from vibrav.molcas import Output
from exa.util.units import Time, Length
from exa.util.constants import (speed_of_light_in_vacuum as speed_of_light,
                                Planck_constant as planck_constant)
from exa.util import conversions as conv
from vibrav.numerical.vibronic_func import *
from vibrav.core.config import Config
from vibrav.numerical.degeneracy import energetic_degeneracy
from vibrav.numerical.boltzmann import boltz_dist
from vibrav.util.open_files import open_txt
from vibrav.util.math import get_triu, ishermitian, isantihermitian, abs2
from vibrav.util.print import dataframe_to_txt
from glob import glob
from datetime import datetime, timedelta
from time import time

class Vibronic:
    '''
    Main class to run vibronic coupling calculations.

    Required arguments in configuration file.

    +------------------------+--------------------------------------------------+----------------------------+
    | Argument               | Description                                      | Data Type                  |
    +========================+==================================================+============================+
    | number_of_multiplicity | Number of multiplicities from calculation.       | :obj:`int`                 |
    +------------------------+--------------------------------------------------+----------------------------+
    | spin_multiplicity      | List of the ordering of the spin multiplicities. | :obj:`tuple` of :obj:`int` |
    |                        | Must be in the same order as was done in the     |                            |
    |                        | calculation.                                     |                            |
    +------------------------+--------------------------------------------------+----------------------------+
    | number_of_states       | Number of states in each multiplicity.           | :obj:`tuple` of :obj:`int` |
    +------------------------+--------------------------------------------------+----------------------------+
    | number_of_nuclei       | Number of nuclei in the system.                  | :obj:`int`                 |
    +------------------------+--------------------------------------------------+----------------------------+
    | number_of_modes        | Number of normal modes in the molecule.          | :obj:`int`                 |
    +------------------------+--------------------------------------------------+----------------------------+
    | zero_order_file        | Filepath of the calculation at the equilibrium   | :obj:`str`                 |
    |                        | coordinate. Must contain the spin-free property  |                            |
    |                        | of interest.                                     |                            |
    +------------------------+--------------------------------------------------+----------------------------+
    | oscillator_spin_states | Number of oscillators to calculate from the      | :obj:`int`                 |
    |                        | ground state.                                    |                            |
    +------------------------+--------------------------------------------------+----------------------------+

    Default arguments in configuration file.

    +------------------+------------------------------------------------------------+----------------+
    | Argument         | Description                                                | Default Value  |
    +==================+============================================================+================+
    | sf_energies_file | Filepath of the spin-free energies.                        | ''             |
    +------------------+------------------------------------------------------------+----------------+
    | so_energies_file | Filepath of the spin-orbit energies.                       | ''             |
    +------------------+------------------------------------------------------------+----------------+
    | angmom_file      | Starting string of the angular momentum spin-orbit files.  | angmom         |
    +------------------+------------------------------------------------------------+----------------+
    | dipole_file      | Starting string of the transition dipole moment spin-orbit | dipole         |
    |                  | files.                                                     |                |
    +------------------+------------------------------------------------------------+----------------+
    | quadrupole_file  | Starting string of the transition quadrupole moment        | quadrupole     |
    |                  | spin-orbit files.                                          |                |
    +------------------+------------------------------------------------------------+----------------+
    | degen_delta      | Cut-off parameter for the energy difference in the         | 1e-7 Ha        |
    |                  | denominator for the pertubation theory.                    |                |
    +------------------+------------------------------------------------------------+----------------+
    | eigvectors_file  | Filepath of the spin-orbit eigenvectors.                   | eigvectors.txt |
    +------------------+------------------------------------------------------------+----------------+
    | so_cont_tol      | Cut-off parameter for the minimum spin-free contribution   | None           |
    |                  | to each spin-orbit state.                                  |                |
    +------------------+------------------------------------------------------------+----------------+
    '''
    _required_inputs = {'number_of_multiplicity': int, 'spin_multiplicity': (tuple, int),
                        'number_of_states': (tuple, int), 'number_of_nuclei': int,
                        'number_of_modes': int, 'zero_order_file': str}
    _default_inputs = {'sf_energies_file': ('', str), 'so_energies_file': ('', str),
                       'angmom_file': ('angmom', str), 'dipole_file': ('dipole', str),
                       'spin_file': ('spin', str), 'quadrupole_file': ('quadrupole', str),
                       'degen_delta': (1e-7, float), 'eigvectors_file': ('eigvectors.txt', str),
                       'so_cont_tol': (None, float), 'sparse_hamiltonian': (False, bool),
                       'states': (None, int)}
    @staticmethod
    def check_size(data, size, var_name, dataframe=False):
        '''
        Simple method to check the size of the input array.

        Args:
            data (np.array or pd.DataFrame): Data array to determine the size.
            size (tuple): Size expected.
            var_name (str): Name of the variable for printing in the error message.

        Raises:
            ValueError: If the given array does not have the shape given in the `size` parameter.
            TypeError: If there are any not a number or null values in the array.
        '''
        if data.shape != size:
            # TODO: we raise a ValueError as it is the closest to something that
            #       makes sense to a LengthError
            #       might be a good idea to create a custom LengthError as it is
            #       very important in this class
            raise ValueError("'{var}' is not of proper size, ".format(var=var_name) \
                            +"currently {curr} expected {ex}".format(curr=data.shape, ex=size))
        try:
            _ = np.any(np.isnan(data))
            numpy = True
        except TypeError:
            numpy = False
        if numpy:
            if np.any(np.isnan(data)):
                raise TypeError("NaN values were found in the data for '{}'".format(var_name))
        else:
            if np.any(pd.isnull(data)):
                raise TypeError("NaN values were found in the data for '{}'".format(var_name))

    def _parse_energies(self, ed, sf_file='', so_file=''):
        # parse the energies from the output is the energy files are not available
        if sf_file != '':
            try:
                energies_sf = pd.read_csv(self.config.sf_energies_file, header=None,
                                          comment='#').values.reshape(-1,)
            except FileNotFoundError:
                text = "The file {} was not found. Reading the spin-free energies directly " \
                       +"from the zero order output file {}."
                warnings.warn(text.format(self.config.sf_energies_file, config.zero_order_file),
                              Warning)
                ed.parse_sf_energy()
                energies_sf = ed.sf_energy['energy'].values
        else:
            ed.parse_sf_energy()
            energies_sf = ed.sf_energy['energy'].values
        self.check_size(energies_sf, (self.nstates_sf,), 'energies_sf')
        if so_file != '':
            try:
                energies_so = pd.read_csv(self.config.so_energies_file, header=None,
                                          comment='#').values.reshape(-1,)
            except FileNotFoundError:
                text = "The file {} was not found. Reading the spin-orbit energies directly " \
                       +"from the zero order output file {}."
                warnings.warn(text.format(self.config.so_energies_file, config.zero_order_file),
                              Warning)
                ed.parse_so_energy()
                energies_so = ed.so_energy['energy'].values
        else:
            ed.parse_so_energy()
            energies_so = ed.so_energy['energy'].values
        self.check_size(energies_so, (self.nstates,), 'energies_so')
        return energies_sf, energies_so

    @staticmethod
    def _get_states(energies, states):
        df = pd.DataFrame.from_dict({'energies': energies, 'sdx': range(len(energies))})
        df.sort_values(by=['energies'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        incl_states = np.zeros(df.shape[0])
        incl_states[range(states)] = 1
        incl_states = incl_states.astype(bool)
        df['incl_states'] = incl_states
        df.sort_values(by=['sdx'], inplace=True)
        incl_states = df['incl_states'].values
        return incl_states

    def get_hamiltonian_deriv(self, select_fdx, delta, redmass, nmodes, use_sqrt_rmass,
                              sparse_hamiltonian):
        # read the hamiltonian files in each of the confg??? directories
        # it is assumed that the directories are named confg with a 3-fold padded number (000)
        padding = 3
        plus_matrix = []
        minus_matrix = []
        found_modes = []
        if isinstance(select_fdx, (list, tuple, np.ndarray)):
            if select_fdx[0] == -1 and len(select_fdx) == 1:
                select_fdx = select_fdx[0]
            elif select_fdx[0] != -1:
                pass
            else:
                raise ValueError("The all condition for selecting frequencies (-1) was passed " \
                                +"along with other frequencies.")
        if select_fdx == -1:
            freq_range = list(range(1, nmodes+1))
        else:
            if isinstance(select_fdx, int): select_fdx = [select_fdx]
            freq_range = np.array(select_fdx) + 1
        nselected = len(freq_range)
        for idx in freq_range:
            # error catching serves the purpose to know which
            # of the hamiltonian files are missing
            try:
                plus = open_txt(os.path.join('confg'+str(idx).zfill(padding), 'ham-sf.txt'),
                                fill=sparse_hamiltonian)
                try:
                    minus = open_txt(os.path.join('confg'+str(idx+nmodes).zfill(padding),
                                                  'ham-sf.txt'), fill=sparse_hamiltonian)
                except FileNotFoundError:
                    warnings.warn("Could not find ham-sf.txt file for in directory " \
                                  +'confg'+str(idx+nmodes).zfill(padding) \
                                  +"\nIgnoring frequency index {}".format(idx), Warning)
                    continue
            except FileNotFoundError:
                warnings.warn("Could not find ham-sf.txt file for in directory " \
                              +'confg'+str(idx).zfill(padding) \
                              +"\nIgnoring frequency index {}".format(idx), Warning)
                continue
            # put it all together only if both plus and minus
            # ham-sf.txt files are found
            plus_matrix.append(plus)
            minus_matrix.append(minus)
            found_modes.append(idx-1)
        ham_plus = pd.concat(plus_matrix, ignore_index=True)
        ham_minus = pd.concat(minus_matrix, ignore_index=True)
        dham_dq = ham_plus - ham_minus
        if nselected != len(found_modes):
            warnings.warn("Number of selected normal modes is not equal to found modes, " \
                         +"currently, {} and {}\n".format(nselected, len(found_modes)) \
                         +"Overwriting the number of selceted normal modes by the number "\
                         +"of found modes.", Warning)
            nselected = len(found_modes)
        self.check_size(dham_dq, (self.nstates_sf*nselected, self.nstates_sf), 'dham_dq')
        # TODO: this division by the sqrt of the mass needs to be verified
        #       left as is for the time being as it was in the original code
        sf_sqrt_rmass = np.repeat(np.sqrt(redmass.loc[found_modes].values*(1/conv.amu2u)),
                                  self.nstates_sf).reshape(-1, 1)
        sf_delta = np.repeat(delta.loc[found_modes].values, self.nstates_sf).reshape(-1, 1)
        if use_sqrt_rmass:
            to_dq = 2 * sf_sqrt_rmass * sf_delta
        else:
            warnings.warn("We assume that you used non-mass-weighted displacements to generate " \
                          +"the displaced structures. We cannot ensure that this actually works.",
                          Warning)
            to_dq = 2 * sf_delta
        # convert to normal coordinates
        dham_dq = dham_dq / to_dq
        # add a frequency index reference
        dham_dq['freqdx'] = np.repeat(found_modes, self.nstates_sf)
        return dham_dq

    def magnetic_oscillator(self):
        raise NotImplementedError("Needs to be fixed!!!")
        #config = self.config
        #nstates = self.nstates
        #oscil_states = int(config.oscillator_spin_states)
        #speed_of_light_au = speed_of_light * Length['m', 'au'] / Time['s', 'au']
        ## read in the spin-orbit energies
        #energies_so = pd.read_csv(config.so_energies_file, header=None,
        #                          comment='#').values.reshape(-1,)
        ## read in all of the angular momentum data and check size
        #ang_x = open_txt('angmom-1.txt').values
        #self.check_size(ang_x, (nstates, nstates), 'ang_x')
        #ang_y = open_txt('angmom-2.txt').values
        #self.check_size(ang_y, (nstates, nstates), 'ang_y')
        #ang_z = open_txt('angmom-3.txt').values
        #self.check_size(ang_z, (nstates, nstates), 'ang_z')
        ## read in all of the spin data and check size
        #spin_x = open_txt('spin-1.txt').values
        #self.check_size(spin_x, (nstates, nstates), 'spin_x')
        #spin_y = open_txt('spin-2.txt').values
        #self.check_size(spin_y, (nstates, nstates), 'spin_y')
        #spin_z = open_txt('spin-3.txt').values
        #self.check_size(spin_z, (nstates, nstates), 'spin_z')
        ## allocate data for x y and z components of magnetic dipoles
        #mx = np.zeros((nstates, nstates), dtype=np.complex128)
        #my = np.zeros((nstates, nstates), dtype=np.complex128)
        #mz = np.zeros((nstates, nstates), dtype=np.complex128)
        ## calclulate each element
        #mx = (1/2) * (ang_x*1j + 2*spin_x)
        #my = (1/2) * (ang_y*1j + 2j*spin_y)
        #mz = (1/2) * (ang_z*1j + 2*spin_z)
        ## allocate memory for magnetic dipoles
        #mdip = np.zeros((nstates, nstates), dtype=np.float64)
        ## calculate the magnetic dipoles
        #mdip = self._abs2(mx) + self._abs2(my) + self._abs2(mz)
        ## allocate memory for magnetic oscillator strengths
        #oscil = np.zeros((oscil_states, oscil_states), dtype=np.float64)
        #delta_E = np.zeros((oscil_states, oscil_states), dtype=np.float64)
        ## calculate the magnetic oscillator strengths
        #osc_prefac = (2/3) * (1/speed_of_light_au**2)
        #compute_mag_oscil_str(oscil_states, energies_so, 0, osc_prefac, mx, my, mz, oscil, delta_E)
        ##oscil = (2/3) * (1/speed_of_light_au**2) \
        ##            * np.repeat(ed.soc_energies['e_cm^-1'], nstates).reshape(nstates, nstates) \
        ##            * Energy['cm^-1', 'Ha'] * mdip
        #initial = np.repeat(range(oscil_states), oscil_states)
        #final = np.tile(range(oscil_states), oscil_states)
        #delta_E *= Energy['Ha', 'cm^-1']
        #df = pd.DataFrame.from_dict({'initial': initial, 'final': final,
        #                             'delta_E': delta_E.reshape(-1,),
        #                             'oscillator': oscil.reshape(-1,)})
        #self.mag_oscil = df

    def vibronic_coupling(self, property, write_property=True, write_energy=True, write_oscil=True,
                          print_stdout=True, temp=298, eq_cont=False, verbose=False,
                          use_sqrt_rmass=True, select_fdx=-1, boltz_states=None, boltz_tol=1e-6,
                          write_sf_oscil=False, write_sf_property=False, write_dham_dq=False,
                          write_all_oscil=False):
        '''
        Vibronic coupling method to calculate the vibronic coupling by the equations as given
        in reference *J. Phys. Chem. Lett.* **2018**, 9, 887-894. This code follows a similar structure
        as that from a perl script written by Yonaton Heit and Jochen Autschbach, authors of the
        cited paper.

        Note:
            The script is able to calculate the vibronic contributions to the electric_dipole,
            magnetic_dipole and electric_quadrupole, currently. For more properties please reach out
            through github or via email.

            This will only work with data from Molcas/OpenMolcas.

        Warning:
            The value of the energy difference parameter (`degen_delta` in the configuration file)
            and the spin-free contribution to the spin-orbit states cutoff (`so_cont_tol` in the
            configuration file) can be very important in giving "reasonable"
            vibronic intensities. These values should be adjusted and tested accordingly on a
            per-system basis. **We make no guarantees everything will work out of the box**.

        Args:
            property (:obj:`str`): Property of interest to calculate.
            write_property (:obj:`bool`, optional): Write the calculated vibronic property values to file.
                                                    Defaults to `True`.
            write_energy (:obj:`bool`, optional): Write the vibronic energies to file.
                                                  Defaults to `True`.
            write_oscil (:obj:`bool`, optional): Write the vibronic oscillators to file.
                                                 Defaults to `True`.
            print_stdout (:obj:`bool`, optional): Print the progress of the script to stdout.
                                                  Defaults to `True`.
            temp (:obj:`float`, optional): Temperature for the boltzmann statistics. Defaults to 298.
            verbose (:obj:`bool`, optional): Send all availble print statements listing where the
                            program is in the calculation to stdout and timings. Recommended if
                            you have a system with many spin-orbit states. Defaults to `False`.
            use_sqrt_rmass (:obj:`bool`, optional): The calculations used mass-weighted normal modes
                                                    for the displaced structures. This should always
                                                    be the case. Defaults to `True`.
            select_fdx (:obj:`list`, optional): Only use select normal modes in the vibronic coupling
                                                calculation. Defaults to `-1` (all normal modes).
            boltz_states (:obj:`int`, optional): Boltzmann states to calculate in the distribution.
                                                 Defaults to `None` (all states with a distribution
                                                 less than the `boltz_tol` value for the lowest
                                                 frequency).
            boltz_tol (:obj:`float`, optional): Tolerance value for the Boltzmann distribution cutoff.
                                                Defaults to `1e-5`.
            write_sf_oscil (:obj:`bool`, optional): Write the spin-free vibronic oscillators.
                                                    Defaults to `False`.
            write_sf_property (:obj:`bool`, optional): Write the spin-free vibronic property values.
                                                       Defaults to `False`.
            write_dham_dq (:obj:`bool`, optional): Write the hamiltonian derivatives for each normal
                                                   mode. Defaults to `False`.
            write_all_oscil (:obj:`bool`, optional): Write the entire matrix of the vibronic
                                                     oscillator values instead of only those that
                                                     are physically meaningful (positive energy and
                                                     oscillator value). Defaults to `False`.

        Raises:
            NotImplementedError: When the property requested with the `property` parameter does not
                                 have any output parser or just has not been coded yet.
            ValueError: If the array that is expected to be Hermitian actually is not.
        '''
        sparse = True
        store_gs_degen = True
        # for program running ststistics
        program_start = time()
        # to reduce typing
        nstates = self.nstates
        nstates_sf = self.nstates_sf
        config = self.config
        # create the vibronic-outputs directory if not available
        vib_dir = 'vibronic-outputs'
        if not os.path.exists(vib_dir):
            os.mkdir(vib_dir)
        # print out the contents of the config file so the user knows how the parameters were read
        if print_stdout:
            print("Printing contents of config file")
            print("*"*46)
            print(config.to_string())
            print("*"*46)
        # for defining the sizes of the arrays later on
        #oscil_states = int(config.oscillator_spin_states)
        # define constants used later on
        speed_of_light_au = speed_of_light*Length['m', 'au']/Time['s', 'au']
        planck_constant_au = 2*np.pi
        # TODO: these hardcoded values need to be generalized
        # this was used in the old script but needs to be fixed
        fc = 1
        # read all of the data files
        delta = pd.read_csv(config.delta_file, header=None)
        rmass = pd.read_csv(config.reduced_mass_file, header=None)
        freq = pd.read_csv(config.frequency_file, header=None).values.reshape(-1,)
        nmodes = config.number_of_modes
        # calculate the boltzmann factors
        boltz_factor = boltz_dist(freq, temp, boltz_tol, boltz_states)
        cols = boltz_factor.columns.tolist()[:-3]
        boltz = np.zeros((boltz_factor.shape[0], 2))
        # sum the boltzmann factors as we will do the sme thing later on anyway
        # important when looking at the oscillator strengths
        for freqdx, data in boltz_factor.groupby('freqdx'):
            boltz[freqdx][0] = np.sum([val*(idx) for idx, val in enumerate(data[cols].values[0])])
            boltz[freqdx][1] = np.sum([val*(idx+1) for idx, val in enumerate(data[cols[:-1]].values[0])])
        boltz = pd.DataFrame(boltz, columns=['minus', 'plus'])
        boltz['freqdx'] = boltz_factor['freqdx']
        boltz['partition'] = boltz_factor['partition']
        boltz.index = boltz['freqdx'].values
        filename = os.path.join(vib_dir, 'boltzmann-populations.csv')
        boltz.to_csv(filename, index=False)
        if print_stdout and False:
            tmp = boltz_factor.copy()
            tmp = tmp[tmp.columns[:-3]]
            tmp = tmp.T
            tmp.index = pd.Index(range(tmp.shape[0]), name='state')
            tmp.columns = boltz_factor['freqdx']
            print_cols = 6
            text = " Printing Boltzmann populations for each normal mode with the\n" \
                  +" energies from the {} file.".format(config.frequency_file)
            print('='*78)
            print(text)
            print('-'*78)
            print(dataframe_to_txt(tmp, float_format=['{:11.7f}'.format]*nmodes,
                                   ncols=print_cols))
            print('='*78)
            tmp = boltz.copy()
            tmp.index = tmp['freqdx']
            tmp.drop(['freqdx'], inplace=True, axis=1)
            tmp = tmp.T
            print('\n\n')
            text = " Printing the Boltzmann weighting for the plus an minus displaced\n" \
                  +" and the respective partition function for each normal mode."
            print('='*81)
            print(text)
            print('-'*81)
            print(dataframe_to_txt(tmp, float_format=['{:11.7f}'.format]*nmodes,
                                   ncols=print_cols))
            print('='*81)
            #print('-'*80)
            #print("Printing the boltzmann distribution for all")
            #print("of the available frequencies at a temperature: {:.2f}".format(temp))
            #formatters = ['{:.7f}'.format, '{:.7f}'.format, '{:d}'.format, '{:.7f}'.format]
            #print(boltz.to_string(index=False, formatters=formatters))
            #print('-'*80)
            #raise
        # read the dipoles in the zero order file
        # make a multiplicity array for extending the derivative arrays from spin-free
        # states to spin-orbit states
        multiplicity = []
        for idx, mult in enumerate(config.spin_multiplicity):
            multiplicity.append(np.repeat(int(mult), int(config.number_of_states[idx])))
        multiplicity = np.concatenate(tuple(multiplicity))
        self.check_size(multiplicity, (nstates_sf,), 'multiplicity')
        # read the eigvectors data
        eigvectors = open_txt(config.eigvectors_file).values
        if config.so_cont_tol is not None:
            conts = abs2(eigvectors)
            so_cont_limit = conts < config.so_cont_tol
            eigvectors[so_cont_limit] = 0.0
            conts = abs2(eigvectors)
            if print_stdout:
                print("*"*50)
                print("Printing out sum of the percent contribution\n" \
                      +"of each spin-orbit state after removing those\n" \
                      +"less than {}".format(config.so_cont_tol))
                print("*"*50)
                print("Printing sorted and unsorted contributions.")
                print("*"*50)
                unsorted_ser = pd.Series(np.sum(conts, axis=1))
                sorted_ser = unsorted_ser.copy().sort_values()
                df_dict = {'so-index-sorted': sorted_ser.index,
                           'sorted-contributions': sorted_ser.values,
                           'so-index-unsorted': unsorted_ser.index,
                           'unsorted-contributions': unsorted_ser.values}
                df = pd.DataFrame.from_dict(df_dict)
                print(df.to_string(index=False))
        self.check_size(eigvectors, (nstates, nstates), 'eigvectors')
        dham_dq = self.get_hamiltonian_deriv(select_fdx, delta, rmass, nmodes,
                                             use_sqrt_rmass, config.sparse_hamiltonian)
        found_modes = dham_dq['freqdx'].unique()
        # TODO: it would be really cool if we could just input a list of properties to compute
        #       and the program will take care of the rest
        ed = Output(config.zero_order_file)
        # get the property of choice from the zero order file given in the config file
        # the extra column in each of the parsed properties comes from the component column
        # in the molcas output parser
        if property.replace('_', '-') == 'electric-dipole':
            ed.parse_sf_dipole_moment()
            self.check_size(ed.sf_dipole_moment, (nstates_sf*3, nstates_sf+1), 'sf_dipole_moment')
            grouped_data = ed.sf_dipole_moment.groupby('component')
            out_file = 'dipole'
            so_file = config.dipole_file
            idx_map = {1: 'x', 2: 'y', 3: 'z'}
        elif property.replace('_', '-') == 'electric-quadrupole':
            ed.parse_sf_quadrupole_moment()
            self.check_size(ed.sf_quadrupole_moment, (nstates_sf*6, nstates_sf+1),
                             'sf_quadrupole_moment')
            grouped_data = ed.sf_quadrupole_moment.groupby('component')
            out_file = 'quadrupole'
            so_file = config.quadrupole_file
            idx_map = {1: 'xx', 2: 'xy', 3: 'xz', 4: 'yy', 5: 'yz', 6: 'zz'}
        elif property.replace('_', '-') == 'magnetic-dipole':
            ed.parse_sf_angmom()
            self.check_size(ed.sf_angmom, (nstates_sf*3, nstates_sf+1), 'sf_angmom')
            grouped_data = ed.sf_angmom.groupby('component')
            out_file = 'angmom'
            so_file = config.angmom_file
            idx_map = {1: 'x', 2: 'y', 3: 'z'}
        else:
            raise NotImplementedError("Sorry the attribute that you are trying to use is not " \
                                     +"yet implemented.")
        if eq_cont:
            # get the spin-orbit property from the molcas output for the equilibrium geometry
            dfs = []
            for file in glob(so_file+'-?.txt'):
                idx = int(file.split('-')[-1].replace('.txt', ''))
                df = open_txt(file)
                # use a mapper as we cannot ensure that the files are found in any
                # expected order
                df['component'] = idx_map[idx]
                dfs.append(df)
            so_props = pd.concat(dfs, ignore_index=True)
        # number of components
        ncomp = len(idx_map.keys())
        # for easier access
        idx_map_rev = {v: k for k, v in idx_map.items()}
        energies_sf, energies_so = self._parse_energies(ed, config.sf_energies_file,
                                                        config.so_energies_file)
        if config.states is not None:
            incl_states = self._get_states(energies_sf, config.states)
        else:
            incl_states = None
        # timing things
        time_setup = time() - program_start
        # counter just for timing statistics
        vib_times = []
        grouped = dham_dq.groupby('freqdx')
        iter_times = []
        prefactor = []
        degeneracy = energetic_degeneracy(energies_so, config.degen_delta)
        gs_degeneracy = degeneracy.loc[0, 'degen']
        if print_stdout:
            print("--------------------------------------------")
            print("Spin orbit ground state was found to be: {:3d}".format(gs_degeneracy))
            print("--------------------------------------------")
        if store_gs_degen: self.gs_degeneracy = gs_degeneracy
        # initialize the oscillator files
        if write_oscil and property.replace('_', '-') == 'electric-dipole':
            osc_tmp = 'oscillators-{}.txt'.format
            header = "{:>5s} {:>5s} {:>24s} {:>24s} {:>6s} {:>7s}".format
            oscil_formatters = ['{:>5d}'.format]*2+['{:>24.16E}'.format]*2 \
                               +['{:>6d}'.format, '{:>7s}'.format]
            with open(os.path.join(vib_dir, osc_tmp(0)), 'w') as fn:
                fn.write(header('#NROW', 'NCOL', 'OSCIL', 'ENERGY', 'FREQDX', 'SIGN'))
            with open(os.path.join(vib_dir, osc_tmp(1)), 'w') as fn:
                fn.write(header('#NROW', 'NCOL', 'OSCIL', 'ENERGY', 'FREQDX', 'SIGN'))
            with open(os.path.join(vib_dir, osc_tmp(2)), 'w') as fn:
                fn.write(header('#NROW', 'NCOL', 'OSCIL', 'ENERGY', 'FREQDX', 'SIGN'))
            with open(os.path.join(vib_dir, osc_tmp(3)), 'w') as fn:
                fn.write(header('#NROW', 'NCOL', 'OSCIL', 'ENERGY', 'FREQDX', 'SIGN'))
        if write_sf_oscil and property.replace('_', '-') == 'electric-dipole':
            osc_tmp = 'oscillators-sf-{}.txt'.format
            header = "{:>5s} {:>5s} {:>24s} {:>24s} {:>6s} {:>7s}".format
            oscil_formatters = ['{:>5d}'.format]*2+['{:>24.16E}'.format]*2 \
                               +['{:>6d}'.format, '{:>7s}'.format]
            with open(os.path.join(vib_dir, osc_tmp(0)), 'w') as fn:
                fn.write(header('#NROW', 'NCOL', 'OSCIL', 'ENERGY', 'FREQDX', 'SIGN'))
            with open(os.path.join(vib_dir, osc_tmp(1)), 'w') as fn:
                fn.write(header('#NROW', 'NCOL', 'OSCIL', 'ENERGY', 'FREQDX', 'SIGN'))
            with open(os.path.join(vib_dir, osc_tmp(2)), 'w') as fn:
                fn.write(header('#NROW', 'NCOL', 'OSCIL', 'ENERGY', 'FREQDX', 'SIGN'))
            with open(os.path.join(vib_dir, osc_tmp(3)), 'w') as fn:
                fn.write(header('#NROW', 'NCOL', 'OSCIL', 'ENERGY', 'FREQDX', 'SIGN'))
        for fdx, founddx in enumerate(found_modes):
            vib_prop = np.zeros((2, ncomp, nstates, nstates), dtype=np.complex128)
            vib_prop_sf = np.zeros((2, ncomp, nstates_sf, nstates_sf), dtype=np.float64)
            vib_prop_sf_so_len = np.zeros((2, ncomp, nstates, nstates), dtype=np.float64)
            vib_start = time()
            if print_stdout:
                print("*******************************************")
                print("*     RUNNING VIBRATIONAL MODE: {:5d}     *".format(founddx+1))
                print("*******************************************")
            # assume that the hamiltonian values are real which they should be anyway
            dham_dq_mode = np.real(grouped.get_group(founddx).drop('freqdx', axis=1).values)
            self.check_size(dham_dq_mode, (nstates_sf, nstates_sf), 'dham_dq_mode')
            tdm_prefac = np.sqrt(planck_constant_au \
                                 /(2*speed_of_light_au*freq[founddx]/Length['cm', 'au']))/(2*np.pi)
            print("TDM prefac: {:.4f}".format(tdm_prefac))
            prefactor.append(tdm_prefac)
            # iterate over all of the available components
            for idx, (key, val) in enumerate(grouped_data):
                start = time()
                # get the values of the specific component
                prop = val.drop('component', axis=1).values
                self.check_size(prop, (nstates_sf, nstates_sf), 'prop_{}'.format(key))
                # initialize arrays
                # spin-free derivatives
                dprop_dq_sf = np.zeros((nstates_sf, nstates_sf), dtype=np.float64)
                # spin-free derivatives extended into the number of spin-orbit states
                # this gets the array ready for spin-orbit mixing
                dprop_dq_so = np.zeros((nstates, nstates), dtype=np.float64)
                # spin-orbit derivatives
                dprop_dq = np.zeros((nstates, nstates), dtype=np.complex128)
                # calculate everything
                compute_d_dq_sf(nstates_sf, dham_dq_mode, prop, energies_sf, dprop_dq_sf,
                                config.degen_delta, incl_states=incl_states)
                sf_to_so(nstates_sf, nstates, multiplicity, dprop_dq_sf, dprop_dq_so)
                compute_d_dq(nstates, eigvectors, dprop_dq_so, dprop_dq)
                # check if the array is hermitian
                if property == 'electric_dipole':
                    if not ishermitian(dprop_dq):
                        text = "The vibronic electric dipole at frequency {} for component {} " \
                               +"was not found to be hermitian."
                        raise ValueError(text.format(fdx, key))
                    if not ishermitian(dprop_dq_sf):
                        text = "The vibronic electric dipole at frequency {} for component {} " \
                               +"was not found to be hermitian."
                        raise ValueError(text.format(fdx, key))
                elif property == 'magnetic_dipole':
                    if not isantihermitian(dprop_dq):
                        text = "The vibronic magentic dipole at frequency {} for component {} " \
                               +"was not found to be non-hermitian."
                        raise ValueError(text.format(fdx, key))
                elif property == 'electric_quadrupole':
                    if not ishermitian(dprop_dq):
                        text = "The vibronic electric quadrupole at frequency {} for " \
                               +"component {} was not found to be hermitian."
                        raise ValueError(text.format(fdx, key))
                dprop_dq *= tdm_prefac
                dprop_dq_sf *= tdm_prefac
                dprop_dq_so *= tdm_prefac
                # generate the full property vibronic states following equation S3 for the reference
                if eq_cont:
                    so_prop = so_props.groupby('component').get_group(key).drop('component', axis=1)
                    vib_prop_plus = fc*(so_prop + dprop_dq)
                    vib_prop_minus = fc*(so_prop - dprop_dq)
                else:
                    # store the transpose as it will make some things easier down the line
                    vib_prop_plus = fc*dprop_dq.T
                    vib_prop_minus = fc*-dprop_dq.T
                    vib_prop_sf_plus = fc*dprop_dq_sf.T
                    vib_prop_sf_minus = fc*-dprop_dq_sf.T
                    vib_prop_sf_so_len_plus = fc*dprop_dq_so.T
                    vib_prop_sf_so_len_minus = fc*-dprop_dq_so.T
                # store in array
                vib_prop[0][idx_map_rev[key]-1] = vib_prop_minus
                vib_prop[1][idx_map_rev[key]-1] = vib_prop_plus
                vib_prop_sf[0][idx_map_rev[key]-1] = vib_prop_sf_minus
                vib_prop_sf[1][idx_map_rev[key]-1] = vib_prop_sf_plus
                vib_prop_sf_so_len[0][idx_map_rev[key]-1] = vib_prop_sf_so_len_minus
                vib_prop_sf_so_len[1][idx_map_rev[key]-1] = vib_prop_sf_so_len_plus
            # calculate the oscillator strengths
            evib = freq[founddx]*conv.inv_m2Ha*100
            initial = np.tile(range(nstates), nstates)+1
            final = np.repeat(range(nstates), nstates)+1
            template = "{:6d}  {:6d}  {:>18.9E}  {:>18.9E}\n".format
            if write_property:
                for idx, (minus, plus) in enumerate(zip(*vib_prop)):
                    plus_T = plus.flatten()
                    real = np.real(plus_T)
                    imag = np.imag(plus_T)
                    dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'plus')
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name, 0o755, exist_ok=True)
                    filename = os.path.join(dir_name, out_file+'-{}.txt'.format(idx+1))
                    with open(filename, 'w') as fn:
                        fn.write('{:>5s}  {:>6s}  {:>18s}  {:>18s}\n'.format('#NROW', 'NCOL',
                                                                             'REAL', 'IMAG'))
                        for i in range(nstates*nstates):
                            fn.write(template(initial[i], final[i], real[i], imag[i]))
                    minus_T = minus.flatten()
                    real = np.real(minus_T)
                    imag = np.imag(minus_T)
                    dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'minus')
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name, 0o755)
                    filename = os.path.join(dir_name, out_file+'-{}.txt'.format(idx+1))
                    with open(filename, 'w') as fn:
                        fn.write('{:>5s}  {:>6s}  {:>10s}  {:>10s}\n'.format('#NROW', 'NCOL',
                                                                             'REAL', 'IMAG'))
                        for i in range(nstates*nstates):
                            fn.write(template(initial[i], final[i], real[i], imag[i]))
                dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'minus')
                with open(os.path.join(dir_name, 'energies.txt'), 'w') as fn:
                    fn.write('# {} (atomic units)\n'.format(nstates))
                    energies = energies_so + (1./2.)*evib - energies_so[0]
                    energies[range(gs_degeneracy)] = energies_so[:gs_degeneracy] \
                                                        - energies_so[0] + (3./2.)*evib
                    for energy in energies:
                        fn.write('{:.9E}\n'.format(energy))
                dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'plus')
                with open(os.path.join(dir_name, 'energies.txt'), 'w') as fn:
                    fn.write('# {} (atomic units)\n'.format(nstates))
                    energies = energies_so + (3./2.)*evib - energies_so[0]
                    energies[range(gs_degeneracy)] = energies_so[:gs_degeneracy] \
                                                        - energies_so[0] + (1./2.)*evib
                    for energy in energies:
                        fn.write('{:.9E}\n'.format(energy))
            if write_sf_property:
                initial = np.tile(range(nstates_sf), nstates_sf)+1
                final = np.repeat(range(nstates_sf), nstates_sf)+1
                for idx, (minus, plus) in enumerate(zip(*vib_prop_sf)):
                    plus_T = plus.flatten()
                    real = np.real(plus_T)
                    imag = np.imag(plus_T)
                    dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'plus')
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name, 0o755, exist_ok=True)
                    filename = os.path.join(dir_name, out_file+'-sf-{}.txt'.format(idx+1))
                    with open(filename, 'w') as fn:
                        fn.write('{:>5s}  {:>6s}  {:>18s}  {:>18s}\n'.format('#NROW', 'NCOL',
                                                                             'REAL', 'IMAG'))
                        for i in range(nstates_sf*nstates_sf):
                            fn.write(template(initial[i], final[i], real[i], imag[i]))
                    minus_T = minus.flatten()
                    real = np.real(minus_T)
                    imag = np.imag(minus_T)
                    dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'minus')
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name, 0o755)
                    filename = os.path.join(dir_name, out_file+'-sf-{}.txt'.format(idx+1))
                    with open(filename, 'w') as fn:
                        fn.write('{:>5s}  {:>6s}  {:>10s}  {:>10s}\n'.format('#NROW', 'NCOL',
                                                                             'REAL', 'IMAG'))
                        for i in range(nstates_sf*nstates_sf):
                            fn.write(template(initial[i], final[i], real[i], imag[i]))
            if write_sf_property:
                initial = np.tile(range(nstates), nstates)+1
                final = np.repeat(range(nstates), nstates)+1
                for idx, (minus, plus) in enumerate(zip(*vib_prop_sf_so_len)):
                    plus_T = plus.flatten()
                    real = np.real(plus_T)
                    imag = np.imag(plus_T)
                    dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'plus')
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name, 0o755, exist_ok=True)
                    filename = os.path.join(dir_name, out_file+'-sf-so-len-{}.txt'.format(idx+1))
                    with open(filename, 'w') as fn:
                        fn.write('{:>5s}  {:>6s}  {:>18s}  {:>18s}\n'.format('#NROW', 'NCOL',
                                                                             'REAL', 'IMAG'))
                        for i in range(nstates*nstates):
                            fn.write(template(initial[i], final[i], real[i], imag[i]))
                    minus_T = minus.flatten()
                    real = np.real(minus_T)
                    imag = np.imag(minus_T)
                    dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'minus')
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name, 0o755)
                    filename = os.path.join(dir_name, out_file+'-sf-so-len-{}.txt'.format(idx+1))
                    with open(filename, 'w') as fn:
                        fn.write('{:>5s}  {:>6s}  {:>10s}  {:>10s}\n'.format('#NROW', 'NCOL',
                                                                             'REAL', 'IMAG'))
                        for i in range(nstates*nstates):
                            fn.write(template(initial[i], final[i], real[i], imag[i]))
            if write_dham_dq:
                initial = np.tile(range(nstates_sf), nstates_sf)+1
                final = np.repeat(range(nstates_sf), nstates_sf)+1
                dir_name = os.path.join('vib'+str(founddx+1).zfill(3))
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, 0o755)
                filename = os.path.join(dir_name, 'hamiltonian-derivs.txt')
                real = np.real(dham_dq_mode.flatten(order='F'))
                imag = np.imag(dham_dq_mode.flatten(order='F'))
                with open(filename, 'w') as fn:
                    fn.write('{:>5s}  {:>6s}  {:>10s}  {:>10s}\n'.format('#NROW', 'NCOL',
                                                                         'REAL', 'IMAG'))
                    for i in range(nstates_sf*nstates_sf):
                        fn.write(template(initial[i], final[i], real[i], imag[i]))
            if (property.replace('_', '-') == 'electric-dipole') and write_oscil:
                mapper = {0: 'iso', 1: 'x', 2: 'y', 3: 'z'}
                # finally get the oscillator strengths from equation S12
                to_drop = ['component', 'freqdx', 'sign', 'prop']
                nrow = np.tile(range(nstates), nstates) + 1
                ncol = np.repeat(range(nstates), nstates) + 1
                for idx, (val, sign) in enumerate(zip([-1, 1], ['minus', 'plus'])):
                    boltz_factor = boltz.loc[founddx, sign]
                    absorption = abs2(vib_prop[idx].reshape(ncomp, nstates*nstates))
                    # get the transition energies
                    energy = energies_so.reshape(-1, 1) - energies_so.reshape(-1,) + val*evib
                    energy = energy.flatten()
                    # check for correct size
                    self.check_size(energy, (nstates*nstates,), 'energy')
                    self.check_size(absorption, (ncomp, nstates*nstates), 'absorption')
                    # compute the isotropic oscillators
                    oscil = boltz_factor * 2./3. * compute_oscil_str(np.sum(absorption, axis=0),
                                                                     energy)
                    # write to file
                    template = ' '.join(['{:>5d}']*2 + ['{:>24.16E}']*2 \
                                        + ['{:>6d}', '{:>7s}'])
                    filename = os.path.join('vibronic-outputs', 'oscillators-0.txt')
                    start = time()
                    with open(filename, 'a') as fn:
                        text = ''
                        # use a for loop instead of a df.to_string() as it is significantly faster
                        for nr, nc, osc, eng in zip(nrow, ncol, oscil, energy):
                            if not write_all_oscil:
                                if osc > 0 and eng > 0:
                                    text += '\n'+template.format(nr, nc, osc, eng, founddx, sign)
                            else:
                                text += '\n'+template.format(nr, nc, osc, eng, founddx, sign)
                        fn.write(text)
                    if print_stdout:
                        text = " Wrote isotropic oscillators to {} for sign {} in {:.2f} s"
                        print(text.format(filename, sign, time() - start))
                    # compute the oscillators for the individual cartesian components
                    for idx, component in enumerate(absorption):
                        self.check_size(component, (nstates*nstates,),
                                        'absorption component {}'.format(idx))
                        oscil = boltz_factor * 2. * compute_oscil_str(component, energy)
                        filename = os.path.join('vibronic-outputs',
                                                'oscillators-{}.txt'.format(idx+1))
                        start = time()
                        with open(filename, 'a') as fn:
                            text = ''
                            for nr, nc, osc, eng in zip(nrow, ncol, oscil, energy):
                                if not write_all_oscil:
                                    if osc > 0 and eng > 0:
                                        text += '\n'+template.format(nr, nc, osc, eng, founddx,
                                                                     sign)
                                else:
                                    text += '\n'+template.format(nr, nc, osc, eng, founddx, sign)
                            fn.write(text)
                        if print_stdout:
                            text = " Wrote oscillators for {} component to {} for sign " \
                                   +"{} in {:.2f} s"
                            print(text.format(mapper[idx+1], filename, sign, time() - start))
            if (property.replace('_', '-') == 'electric-dipole') and write_sf_oscil:
                mapper = {0: 'iso', 1: 'x', 2: 'y', 3: 'z'}
                # finally get the oscillator strengths from equation S12
                to_drop = ['component', 'freqdx', 'sign', 'prop']
                nrow = np.tile(range(nstates_sf), nstates_sf) + 1
                ncol = np.repeat(range(nstates_sf), nstates_sf) + 1
                for idx, (val, sign) in enumerate(zip([-1, 1], ['minus', 'plus'])):
                    boltz_factor = boltz.loc[founddx, sign]
                    absorption = abs2(vib_prop_sf[idx].reshape(ncomp, nstates_sf*nstates_sf))
                    # get the transition energies
                    energy = energies_sf.reshape(-1, 1) - energies_sf.reshape(-1,) + val*evib
                    energy = energy.flatten()
                    # check for correct size
                    self.check_size(energy, (nstates_sf*nstates_sf,), 'energy')
                    self.check_size(absorption, (ncomp, nstates_sf*nstates_sf), 'absorption')
                    # compute the isotropic oscillators
                    oscil = boltz_factor * 2./3. * compute_oscil_str(np.sum(absorption, axis=0),
                                                                     energy)
                    # write to file
                    template = ' '.join(['{:>5d}']*2 + ['{:>24.16E}']*2 \
                                        + ['{:>6d}', '{:>7s}'])
                    filename = os.path.join('vibronic-outputs', 'oscillators-sf-0.txt')
                    start = time()
                    with open(filename, 'a') as fn:
                        text = ''
                        # use a for loop instead of a df.to_string() as it is significantly faster
                        for nr, nc, osc, eng in zip(nrow, ncol, oscil, energy):
                            if write_all_oscil:
                                if osc > 0 and eng > 0:
                                    text += '\n'+template.format(nr, nc, osc, eng, founddx, sign)
                            else:
                                text += '\n'+template.format(nr, nc, osc, eng, founddx, sign)
                        fn.write(text)
                    if print_stdout:
                        text = " Wrote isotropic oscillators to {} for sign {} in {:.2f} s"
                        print(text.format(filename, sign, time() - start))
                    # compute the oscillators for the individual cartesian components
                    for idx, component in enumerate(absorption):
                        self.check_size(component, (nstates_sf*nstates_sf,),
                                        'absorption component {}'.format(idx))
                        oscil = boltz_factor * 2. * compute_oscil_str(component, energy)
                        filename = os.path.join('vibronic-outputs',
                                                'oscillators-sf-{}.txt'.format(idx+1))
                        start = time()
                        with open(filename, 'a') as fn:
                            text = ''
                            for nr, nc, osc, eng in zip(nrow, ncol, oscil, energy):
                                if write_all_oscil:
                                    if osc > 0 and eng > 0:
                                        text += '\n'+template.format(nr, nc, osc, eng, founddx,
                                                                     sign)
                                else:
                                    text += '\n'+template.format(nr, nc, osc, eng, founddx, sign)
                            fn.write(text)
                        if print_stdout:
                            text = " Wrote oscillators for {} component to {} for sign " \
                                   +"{} in {:.2f} s"
                            print(text.format(mapper[idx+1], filename, sign, time() - start))
        if print_stdout:
            print("Writing out the prefactors used for the transition dipole moments.")
        with open(os.path.join(vib_dir, 'alpha.txt'), 'w') as fn:
            fn.write('alpha\n')
            for val in prefactor:
                fn.write('{:.9f}\n'.format(val))
        #program_end = time()
        #if print_stdout:
        #    program_exec = timedelta(seconds=round(program_end - program_start, 0))
        #    vib_time = timedelta(seconds=round(np.mean(vib_times), 0))
        #    setup = timedelta(seconds=round(time_setup, 0))
        #    if write_property:
        #        prop_io_time = timedelta(seconds=round(end_write_prop - start_write_prop, 0))
        #    print("***************************************")
        #    print("* Timing statistics:                  *")
        #    print("* Program Execution:{:.>17s} *".format(str(program_exec)))
        #    print("* Avg. Vib Execution:{:.>16s} *".format(str(vib_time)))
        #    print("* Setup time:{:.>24s} *".format(str(setup)))
        #    if write_property:
        #        print("* I/O:                                *")
        #        print("* Write property:{:.>20s} *".format(str(prop_io_time)))
        #    print("***************************************")

    def __init__(self, config_file, *args, **kwargs):
        config = Config.open_config(config_file, self._required_inputs,
                                    defaults=self._default_inputs)
        # check that the number of multiplicities and states are the same
        if len(config.spin_multiplicity) != len(config.number_of_states):
            print(config.spin_multiplicity, config.number_of_states)
            raise ValueError("Length mismatch of SPIN_MULTIPLICITY " \
                             +"({}) ".format(len(config.spin_multiplicity)) \
                             +"and NUMBER_OF_STATES ({})".format(len(config.number_of_states)))
        if len(config.spin_multiplicity) != int(config.number_of_multiplicity):
            print(config.spin_multiplicity, config.number_of_multiplicity)
            raise ValueError("Length of SPIN_MULTIPLICITY ({}) ".format(config.spin_multiplicity) \
                             +"does not equal the NUMBER_OF_MULTIPLICITY " \
                             +"({})".format(config.number_of_multiplicity))
        nstates = 0
        nstates_sf = 0
        for mult, state in zip(config.spin_multiplicity, config.number_of_states):
            nstates += mult*state
            nstates_sf += state
        self.config = config
        self.nstates = nstates
        self.nstates_sf = nstates_sf

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
        be the order of the multiplicities. **This is to be changed in the
        future**.

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
    from vibrav.util.open_files import open_txt
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
                    dir = os.path.join(path.split(os.sep)[:-1])
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
                    warnings.warn(text.format(dir, idx, Warning))
                    raise FileNotFound
                # read the data
                data = open_txt(file, rearrange=False)
                # ensure that the data has the right things
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
                    # append if there is nothing in the dfs array
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


