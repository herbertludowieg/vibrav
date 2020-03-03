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
from datetime import datetime, timedelta
from time import time
from vibrav.molcas import Output
from exa.util.units import Time, Mass, Energy, Length
from exa.util.constants import (speed_of_light_in_vacuum as speed_of_light,
                                Planck_constant as planck_constant,
                                Boltzmann_constant as boltz_constant)
from vibrav.numerical.vibronic_func import *
from vibrav.core.config import Config
from glob import glob
from vibrav.util.open_files import open_txt
from vibrav.util.math import get_triu, ishermitian

class Vibronic:
    _required_inputs = {'number_of_multiplicity': int, 'spin_multiplicity': (tuple, int),
                        'number_of_states': (tuple, int), 'number_of_nuclei': int,
                        'number_of_modes': int, 'zero_order_file': str,
                        'oscillator_spin_states': int}
    _default_inputs = {'sf_energies_file': None, 'so_energies_file': None, 'angmom_file': 'angmom',
                       'dipole_file': 'dipole', 'spin_file': 'spin', 'quadrupole_file': 'quadrupole',
                       'degen_delta': 1e-5}
    @staticmethod
    def _check_size(data, size, var_name, dataframe=False):
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
                raise TypeError("NaN values were found in the data for '{var}'".format(var_name))
        else:
            if np.any(pd.isnull(data)):
                raise TypeError("NaN values were found in the data for '{var}'".format(var_name))

    @staticmethod
    def boltz_factor(energies_so):
        raise NotImplementedError("Coming Soon!!")

    @staticmethod
    def determine_degeneracy(data_df, degen_delta, rtol=1e-12, numpy=True):
        degen_states = []
        idx = 0
        if not numpy:
            sorted = data_df.sort_values()
            index = sorted.index.values
            data = sorted.values
        else:
            df = pd.Series(data_df)
            sorted = df.sort_values()
            index = sorted.index.values
            data = sorted.values
        while idx < data.shape[0]:
            degen = np.isclose(data[idx], data, atol=degen_delta, rtol=rtol)
            ddx = np.where(degen)[0]
            degen_vals = data[ddx]
            degen_index = index[ddx]
            mean = np.mean(degen_vals)
            idx += ddx.shape[0]
            df = pd.DataFrame.from_dict({'values': [mean], 'degen': [ddx.shape[0]]})
            found = np.transpose(degen_index)
            df['index'] = [found]
            degen_states.append(df)
        degeneracy = pd.concat(degen_states, ignore_index=True)
        return degeneracy

    def magnetic_oscillator(self):
        raise NotImplementedError("Needs to be fixed!!!")
        config = self.config
        nstates = self.nstates
        oscil_states = int(config.oscillator_spin_states)
        speed_of_light_au = speed_of_light * Length['m', 'au'] / Time['s', 'au']
        # read in the spin-orbit energies
        energies_so = pd.read_csv(config.so_energies_file, header=None,
                                  comment='#').values.reshape(-1,)
        # read in all of the angular momentum data and check size
        ang_x = open_txt('angmom-1.txt').values
        self._check_size(ang_x, (nstates, nstates), 'ang_x')
        ang_y = open_txt('angmom-2.txt').values
        self._check_size(ang_y, (nstates, nstates), 'ang_y')
        ang_z = open_txt('angmom-3.txt').values
        self._check_size(ang_z, (nstates, nstates), 'ang_z')
        # read in all of the spin data and check size
        spin_x = open_txt('spin-1.txt').values
        self._check_size(spin_x, (nstates, nstates), 'spin_x')
        spin_y = open_txt('spin-2.txt').values
        self._check_size(spin_y, (nstates, nstates), 'spin_y')
        spin_z = open_txt('spin-3.txt').values
        self._check_size(spin_z, (nstates, nstates), 'spin_z')
        # allocate data for x y and z components of magnetic dipoles
        mx = np.zeros((nstates, nstates), dtype=np.complex128)
        my = np.zeros((nstates, nstates), dtype=np.complex128)
        mz = np.zeros((nstates, nstates), dtype=np.complex128)
        # calclulate each element
        mx = (1/2) * (ang_x*1j + 2*spin_x)
        my = (1/2) * (ang_y*1j + 2j*spin_y)
        mz = (1/2) * (ang_z*1j + 2*spin_z)
        # allocate memory for magnetic dipoles
        mdip = np.zeros((nstates, nstates), dtype=np.float64)
        # calculate the magnetic dipoles
        mdip = self._abs2(mx) + self._abs2(my) + self._abs2(mz)
        # allocate memory for magnetic oscillator strengths
        oscil = np.zeros((oscil_states, oscil_states), dtype=np.float64)
        delta_E = np.zeros((oscil_states, oscil_states), dtype=np.float64)
        # calculate the magnetic oscillator strengths
        osc_prefac = (2/3) * (1/speed_of_light_au**2)
        compute_mag_oscil_str(oscil_states, energies_so, 0, osc_prefac, mx, my, mz, oscil, delta_E)
        #oscil = (2/3) * (1/speed_of_light_au**2) \
        #            * np.repeat(ed.soc_energies['e_cm^-1'], nstates).reshape(nstates, nstates) \
        #            * Energy['cm^-1', 'Ha'] * mdip
        initial = np.repeat(range(oscil_states), oscil_states)
        final = np.tile(range(oscil_states), oscil_states)
        delta_E *= Energy['Ha', 'cm^-1']
        df = pd.DataFrame.from_dict({'initial': initial, 'final': final,
                                     'delta_E': delta_E.reshape(-1,),
                                     'oscillator': oscil.reshape(-1,)})
        self.mag_oscil = df

    def vibronic_coupling(self, property, write_property=True, write_energy=True, write_oscil=True,
                          print_stdout=True, temp=298, eq_cont=True, verbose=False, sparse=True,
                          use_sqrt_rmass=True, store_gs_degen=True, select_fdx=-1):
        '''
        Vibronic coupling method to calculate the vibronic coupling by the equations as given
        in reference J. Phys. Chem. Lett. 2018, 9, 887-894. This code follows a similar structure
        as that from a perl script written by Yonaton Heit and Jochen Autschbach, authors of the
        cited paper.

        The script is able to calculate the vibronic contributions to the electric_dipole,
        magnetic_dipole and electric_quadrupole, currently. For more properties please reach out
        through github or via email.

        Note:
            Any files that are written are part of a sparse hermitian matrix as we only store the upper
            triangular part of any matrices. Meaning, if you want to read the '.txt' files written you
            must take both of these into consideration.

        Args:
            property (str): Property of interest to calculate.
            write_property (bool): Write the calculated vibronic property values to file.
                                   Default: `True`.
            write_energy (bool): Write the vibronic energies to file. Default: `True`.
            write_oscil (bool): Write the vibronic oscillators to file. Default: `True`.
            print_stdout (bool): Print the progress of the script to stdout. Default: `True`.
            temp (float): Temperature for the boltzmann statistics. Default: 298.
            no_vibronic (bool): Not yet implemented. Debug.
            verbose (bool): Send all availble print statements listing where the program is in the
                            calculation to stdout and timings. Recommended if you have a system with
                            many spin-orbit states. Default: `False`.

        Raises:
            NotImplementedError: When the property requested with the `property` parameter does not
                                 have any output parser or just has not been coded yet.
            ValueError: If the array that is expected to be Hermitian actually is not.
        '''
        # for program running ststistics
        program_start = time()
        # to reduce typing
        nstates = self.nstates
        nstates_sf = self.nstates_sf
        config = self.config
        # print out the contents of the config file so the user knows how the parameters were read
        if print_stdout:
            print("Printing contents of config file")
            print("*"*46)
            print(config.to_string())
            print("*"*46)
        # for defining the sizes of the arrays later on
        oscil_states = int(config.oscillator_spin_states)
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
        # read the dipoles in the zero order file
        # make a multiplicity array for extending the derivative arrays from spin-free
        # states to spin-orbit states
        multiplicity = []
        for idx, mult in enumerate(config.spin_multiplicity):
            multiplicity.append(np.repeat(int(mult), int(config.number_of_states[idx])))
        multiplicity = np.concatenate(tuple(multiplicity))
        self._check_size(multiplicity, (nstates_sf,), 'multiplicity')
        # read the eigvectors data
        eigvectors = open_txt('eigvectors.txt').values
        self._check_size(eigvectors, (nstates, nstates), 'eigvectors')
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
            nmodes = config.number_of_modes
            freq_range = list(range(1, nmodes+1))
        else:
            if isinstance(select_fdx, int): select_fdx = [select_fdx]
            freq_range = np.array(select_fdx) + 1
            nmodes = config.number_of_modes
        nselected = len(freq_range)
        for idx in freq_range:
            try:
                plus = open_txt(os.path.join('confg'+str(idx).zfill(padding), 'ham-sf.txt'))
                try:
                    minus = open_txt(os.path.join('confg'+str(idx+nmodes).zfill(padding),
                                                  'ham-sf.txt'))
                except FileNotFoundError:
                    print("Could not find ham-sf.txt file for in directory " \
                         +'confg'+str(idx+nmodes).zfill(padding) \
                         +"\nIgnoring frequency index {}".format(idx))
                    continue
            except FileNotFoundError:
                print("Could not find ham-sf.txt file for in directory " \
                     +'confg'+str(idx).zfill(padding) \
                     +"\nIgnoring frequency index {}".format(idx))
                continue
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
        self._check_size(dham_dq, (nstates_sf*nselected, nstates_sf), 'dham_dq')
        # TODO: this division by the sqrt of the mass needs to be verified
        #       left as is for the time being as it was in the original code
        sf_sqrt_rmass = np.repeat(np.sqrt(rmass.loc[found_modes].values*Mass['u', 'au_mass']),
                                  nstates_sf).reshape(-1, 1)
        sf_delta = np.repeat(delta.loc[found_modes].values, nstates_sf).reshape(-1, 1)
        if use_sqrt_rmass:
            to_dq = 2 * sf_sqrt_rmass * sf_delta
        else:
            to_dq = 2 * sf_delta
        # convert to normal coordinates
        dham_dq = dham_dq / to_dq
        # add a frequency index reference
        dham_dq['freqdx'] = np.repeat(found_modes, nstates_sf)
        # TODO
        # TODO: it would be really cool if we could just input a list of properties to compute
        #       and the program will take care of the rest
        # TODO
        ed = Output(config.zero_order_file)
        # parse the energies from the output is the energy files are not available
        if config.sf_energies_file is not None:
            try:
                energies_sf = pd.read_csv(config.sf_energies_file, header=None,
                                          comment='#').values.reshape(-1,)
            except FileNotFoundError:
                print("The file {} was not found. Reading ".format(config.sf_energies_file) \
                     +"spin-free energies direclty from the " \
                     +"zero order output file {}".format(config.zero_order_file))
                ed.parse_sf_energy()
                energies_sf = ed.sf_energy['energy'].values
        else:
            ed.parse_sf_energy()
            energies_sf = ed.sf_energy['energy'].values
        if config.so_energies_file is not None:
            try:
                energies_so = pd.read_csv(config.so_energies_file, header=None,
                                          comment='#').values.reshape(-1,)
            except FileNotFoundError:
                print("The file {} was not found. Reading ".format(config.so_energies_file) \
                     +"spin-orbit energies direclty from the " \
                     +"zero order output file {}".format(config.zero_order_file))
                ed.parse_so_energy()
                energies_so = ed.so_energy['energy'].values
        else:
            ed.parse_so_energy()
            energies_so = ed.so_energy['energy'].values
        # get the property of choice from the zero order file given in the config file
        # the extra column in each of the parsed properties comes from the component column
        # in the molcas output parser
        if property == 'electric_dipole':
            ed.parse_sf_dipole_moment()
            self._check_size(ed.sf_dipole_moment, (nstates_sf*3, nstates_sf+1), 'sf_dipole_moment')
            grouped_data = ed.sf_dipole_moment.groupby('component')
            out_file = 'dipole'
            so_file = config.dipole_file
            idx_map = {1: 'x', 2: 'y', 3: 'z'}
        elif property == 'electric_quadrupole':
            ed.parse_sf_quadrupole_moment()
            self._check_size(ed.sf_quadrupole_moment, (nstates_sf*6, nstates_sf+1),
                             'sf_quadrupole_moment')
            grouped_data = ed.sf_quadrupole_moment.groupby('component')
            out_file = 'quadrupole'
            so_file = config.quadrupole_file
            idx_map = {1: 'xx', 2: 'xy', 3: 'xz', 4: 'yy', 5: 'yz', 6: 'zz'}
        elif property == 'magnetic_dipole':
            ed.parse_sf_angmom()
            self._check_size(ed.sf_angmom, (nstates_sf*3, nstates_sf+1), 'sf_angmom')
            grouped_data = ed.sf_angmom.groupby('component')
            out_file = 'angmom'
            so_file = config.angmom_file
            idx_map = {1: 'x', 2: 'y', 3: 'z'}
        else:
            raise NotImplementedError("Sorry the attribute that you are trying to use is not " \
                                     +"yet implemented.")
        # get the spin-orbit property from the molcas output for the equilibrium geometry
        dfs = []
        for file in glob(so_file+'-?.txt'):
            idx = int(file.split('-')[-1].replace('.txt', ''))
            df = open_txt(file)
            df['component'] = idx_map[idx]
            dfs.append(df)
        so_props = pd.concat(dfs, ignore_index=True)
        # number of components
        ncomp = len(idx_map.keys())
        # for easier access
        idx_map_rev = {v: k for k, v in idx_map.items()}
        # number of elements in upper triangular matrices for each vibronic block including the diagonal
        upper_nelem = int(nstates*(nstates+1)/2)
        # allocate memory for arrays
        oscil = np.zeros((nselected, 3, nstates*nstates), dtype=np.float64)
        delta_E = np.zeros((nselected, 3, nstates*nstates), dtype=np.float64)
        vibronic_prop = np.zeros((nselected, 3, ncomp, nstates*nstates), dtype=np.complex128)
        #oscil = np.zeros((nmodes, 2, upper_nelem), dtype=np.float64)
        #delta_E = np.zeros((nmodes, 2, upper_nelem), dtype=np.float64)
        #vibronic_prop = np.zeros((nmodes, 2, ncomp, upper_nelem), dtype=np.complex128)
        # timing things
        time_setup = time() - program_start
        # counter just for timing statistics
        vib_times = []
        grouped = dham_dq.groupby('freqdx')
        iter_times = []
        degeneracy = self.determine_degeneracy(energies_so, config.degen_delta)
        gs_degeneracy = degeneracy.loc[0, 'degen']
        if store_gs_degen: self.gs_degeneracy = gs_degeneracy
        for fdx, founddx in enumerate(found_modes):
            vib_prop = np.zeros((3, ncomp, nstates, nstates), dtype=np.complex128)
            vib_start = time()
            if print_stdout:
                print("*******************************************")
                print("*     RUNNING VIBRATIONAL MODE: {:5d}     *".format(founddx+1))
                print("*******************************************")
            # assume that the hamiltonian values are real which they should be anyway
            dham_dq_mode = np.real(grouped.get_group(founddx).values[:,:-1])
            tdm_prefac = np.sqrt(planck_constant_au \
                                 /(2*speed_of_light_au*freq[founddx]/Length['cm', 'au']))/(2*np.pi)
            # iterate over all of the available components
            for idx, (key, val) in enumerate(grouped_data):
                start = time()
                # get the values of the specific component
                prop = val.drop('component', axis=1).values
                self._check_size(prop, (nstates_sf, nstates_sf), 'prop_{}'.format(key))
                # initialize arrays
                dprop_dq_sf = np.zeros((nstates_sf, nstates_sf), dtype=np.float64) # spin-free deriv
                dprop_dq_so = np.zeros((nstates, nstates), dtype=np.float64)       # spin-free deriv
                                                                                   # extended to total
                                                                                   # spin-orbit states
                dprop_dq = np.zeros((nstates, nstates), dtype=np.complex128)       # spin-orbit deriv
                # calculate everything
                compute_d_dq_sf(nstates_sf, dham_dq_mode, prop, energies_sf, dprop_dq_sf,
                                1e-5)
                sf_to_so(nstates_sf, nstates, multiplicity, dprop_dq_sf, dprop_dq_so)
                compute_d_dq(nstates, eigvectors, dprop_dq_so, dprop_dq)
                # check if the array is hermitian
                # this one should be
                dprop_dq *= tdm_prefac
                if not ishermitian(dprop_dq) and property == 'electric_dipole':
                    print("Falied at component {} in prop {}".format(key[0], property))
                    raise ValueError("dprop_dq array is not Hermitian when it is expected to be")
                ## reduce to the upper triangular elements
                #dprop_dq = get_triu(dprop_dq)
                #dprop_dq = dprop_dq.flatten()
                # get the spin-orbit data for the specific component
                so_prop = so_props.groupby('component').get_group(key).drop('component', axis=1).values
                #so_prop = so_prop.flatten()
                #_ = so_props.groupby('component').get_group(key).drop('component', axis=1).values
                #so_prop = get_triu(_)
                # apply a 2 state boltzmann distribution
                # TODO: this needs to be carefully checked
                #boltz_denom = 1+np.exp(-freq[fdx]/(boltz_constant*Energy['J', 'cm^-1']*temp))
                #boltz_plus = 1/boltz_denom
                #boltz_minus = np.exp(-freq[fdx]/(boltz_constant*Energy['J', 'cm^-1']*temp))/boltz_denom
                # generate the full property vibronic states following equation S3 for the reference
                if eq_cont:
                    vib_prop_plus = fc*(so_prop + dprop_dq)
                    vib_prop_minus = fc*(so_prop - dprop_dq)
                    vib_prop_none = fc*(so_prop)
                else:
                    vib_prop_plus = fc*dprop_dq
                    vib_prop_minus = fc*-dprop_dq
                    vib_prop_none = np.zeros((nstates, nstates), dtype=np.float64)
                # store in array
                vibronic_prop[fdx][0][idx_map_rev[key]-1] = vib_prop_minus.flatten()
                vibronic_prop[fdx][1][idx_map_rev[key]-1] = vib_prop_none.flatten()
                vibronic_prop[fdx][2][idx_map_rev[key]-1] = vib_prop_plus.flatten()
                vib_prop[0][idx_map_rev[key]-1] = vib_prop_minus
                vib_prop[1][idx_map_rev[key]-1] = vib_prop_none
                vib_prop[2][idx_map_rev[key]-1] = vib_prop_plus
                # fancy timing stuff
                end = time() - start
                iter_times.append(end)
                # make an estimate of how much longer this will run for
                # does not take into account anything beyond the construction of the derivatives
                eta = timedelta(seconds=round(np.mean(iter_times)*(nselected*ncomp-len(iter_times)), 0))
                if print_stdout and verbose:
                    print(" Computed {:3s} component in {:8.1f} s".format(key, end))
                    print(" ETA:{:.>32s}".format(str(eta)))
                    print("-"*37)
            # calculate the oscillator strengths
            evib = planck_constant_au * speed_of_light_au * freq[founddx]/Length['cm', 'au']
            initial = np.repeat(range(nstates), nstates)+1
            final = np.tile(range(nstates), nstates)+1
            template = "{:6d}  {:6d}  {:>18.9E}  {:>18.9E}\n".format
            if write_property:
                for idx, (plus, none, minus) in enumerate(zip(*vib_prop)):
                    plus_T = plus.T.flatten()
                    real = np.real(plus_T)
                    imag = np.imag(plus_T)
                    dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'plus')
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name, 0o755, exist_ok=True)
                    filename = os.path.join(dir_name, out_file+'-{}.txt'.format(idx+1))
                    with open(filename, 'w') as fn:
                        fn.write('#{:>5s}  {:>6s}  {:>18s}  {:>18s}\n'.format('NROW', 'NCOL', 'REAL', 'IMAG'))
                        for i in range(nstates*nstates):
                            fn.write(template(initial[i], final[i], real[i], imag[i]))
                    minus_T = minus.T.flatten()
                    real = np.real(minus_T)
                    imag = np.imag(minus_T)
                    dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'minus')
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name, 0o755)
                    filename = os.path.join(dir_name, out_file+'-{}.txt'.format(idx+1))
                    with open(filename, 'w') as fn:
                        fn.write('#{:>5s}  {:>6s}  {:>10s}  {:>10s}\n'.format('NROW', 'NCOL', 'REAL', 'IMAG'))
                        for i in range(nstates*nstates):
                            fn.write(template(initial[i], final[i], real[i], imag[i]))
                dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'minus')
                with open(os.path.join(dir_name, 'energies.txt'), 'w') as fn:
                    fn.write('# {} (atomic units)\n'.format(nstates))
                    energies = energies_so + (1./2.)*evib - energies_so[0]
                    energies[range(gs_degeneracy)] = (3./2.)*evib
                    for energy in energies:
                        fn.write('{:.9E}\n'.format(energy))
                dir_name = os.path.join('vib'+str(founddx+1).zfill(3), 'plus')
                with open(os.path.join(dir_name, 'energies.txt'), 'w') as fn:
                    fn.write('# {} (atomic units)\n'.format(nstates))
                    energies = energies_so + (3./2.)*evib - energies_so[0]
                    energies[range(gs_degeneracy)] = (1./2.)*evib
                    for energy in energies:
                        fn.write('{:.9E}\n'.format(energy))
            signs = ['minus', 'none', 'plus']
            if (property == 'electric_dipole' or property == 'magnetic_dipole') and write_oscil:
                if print_stdout and verbose:
                    print(" Computing the oscillator strengths")
                    print("-----------------------------------")
                # finally get the oscillator strengths from equation S12
                to_drop = ['component', 'freqdx', 'sign', 'prop']
                boltz_denom = 1+np.exp(-freq[founddx]/(boltz_constant*Energy['J', 'cm^-1']*temp))
                boltz_plus = 1/boltz_denom
                boltz_minus = np.exp(-freq[founddx]/(boltz_constant*Energy['J', 'cm^-1']*temp))/boltz_denom
                for idx, val in enumerate([-1, 0, 1]):
                    if val == -1:
                        boltz = boltz_minus
                    else:
                        boltz = boltz_plus
                    absorption = np.zeros(nstates*nstates, dtype=np.float64)
                    for component in vibronic_prop[fdx][idx]:
                        absorption += abs2(component)
                    energy = energies_so.reshape(-1,) - energies_so.reshape(-1,1) + val*evib
                    energy = energy.flatten()
                    self._check_size(energy, (nstates*nstates,), 'energy')
                    self._check_size(absorption, (nstates*nstates,), 'absorption')
                    oscil[fdx][idx] = boltz * 2./3. * compute_oscil_str(absorption, energy)
                    delta_E[fdx][idx] = energy
            else:
                write_oscil = False
                for idx, val in enumerate([-1, 1]):
                    energy = energies_so.reshape(-1,) - energies_so.reshape(-1,1) + val*evib
                    energy = energy.flatten()
                    self._check_size(energy, (nstates*nstates,), 'energy')
                    delta_E[fdx][idx] = energy
        # write the values of the vibronic property
        out_dir = 'vibronic-outputs'
        nstates_vib = 2*nselected*nstates
        if False:
            write_energy = True
            template = "{:6d}  {:6d}  {:+.9E}  {:+.9E}\n"
            if print_stdout:
                print('='*68)
                print("Writing vibronic {} to file".format(property))
            if not os.path.exists(out_dir):
                os.mkdir(out_dir, 0o755)
            if sparse:
                # this is an algorithm that will write 2*nmodes blocks of nstates x nstates arrays
                # essentially we have the tinking that only the vibronic levels of the same vibration
                # interact as per the <phi_1|Q_p|phi_2> term in equation S3 in the referenced publication
                # meaning that we can think of generating a sparse matrix of matrices of
                # 2*nmodes x 2*nmodes and each submatrix is nstates x nstates with the diagonal matrices
                # having non-zero values this can be depicted below with 2 normal modes
                # [[[V],[0],[0],[0]],
                #  [[0],[V],[0],[0]],
                #  [[0],[0],[V],[0]],
                #  [[0],[0],[0],[V]]]
                # here V are the vibronic arrays being calculated and 0 are arrays of zeros
                # the vibronic arrays are hermitian by definition and as such we only write the upper
                # triangular elements to file
                # to read the file one must consider that the file is both sparse and hermitian
                # iterate over each component available
                for jdx, (key, val) in enumerate(idx_map_rev.items()):
                    start = time()
                    filename = os.path.join(out_dir, out_file+"-{}.txt")
                    with open(filename.format(val), 'w') as fn:
                        fn.write('#NROW NCOL REAL IMAG\n')
                        # outside of for loops as this will go from 0 to the number of vibronic states
                        final = 0
                        # iterate over normal modes
                        for fdx in range(nmodes):
                            # iterate over the plus and minus vibronic states around the
                            # electronic excitation
                            for idx, sign in enumerate(['minus', 'plus']):
                                # iterate over the columns
                                for i in range(nstates):
                                    # this needs to reset to the initial value after we go to the
                                    # next 'row' in the data
                                    initial = (2*fdx+idx)*nstates
                                    # iterate over the rows
                                    for j in range(nstates):
                                        real = np.real(vibronic_prop[fdx][idx][jdx][j*nstates+i])
                                        imag = np.imag(vibronic_prop[fdx][idx][jdx][j*nstates+i])
                                        fn.write(template.format(initial+1, final+1, real, imag))
                                        initial += 1
                                    final += 1
                        if print_stdout:
                            time_taken = timedelta(seconds=round(time()-start, 0))
                            print("Wrote {:d} lines to {} in {}".format(2*nmodes*nstates*nstates,
                                                                      filename.format(val),
                                                                      str(time_taken)))
            else:
                raise Exception("I will not write a non-sparse matrix....Too many zeros")
                #for jdx, (key, val) in enumerate(idx_map_rev.items()):
                #    time_start = time()
                #    filename = os.path.join(out_dir, out_file+"-{}.txt")
                #    nelem = (nstates_vib)**2
                #    with open(filename.format(val), 'w') as fn:
                #        fn.write('#NROW NCOL REAL IMAG\n')
                #        start = 0
                #        for fdx in range(nmodes):
                #            for idx, sign in enumerate(['minus', 'plus']):
                #                index = 0
                #                for i in range(nstates):
                #                    j = 0
                #                    real = np.real(vibronic_prop[fdx][idx][jdx])
                #                    imag = np.imag(vibronic_prop[fdx][idx][jdx])
                #                    while j < start:
                #                        fn.write(template.format(i+start+1, j+1, 0., 0.))
                #                        j += 1
                #                    while (j >= start and j%nstates < i):
                #                        #fn.write(template.format(i+start+1, j+1, real[index],
                #                        #                         -imag[index]))
                #                        #index += 1
                #                        fn.write(template.format(i+start+1, j+1, 0., 0.))
                #                        j += 1
                #                    while (j < start+nstates and j%nstates >= i):
                #                        fn.write(template.format(i+start+1, j+1, real[index],
                #                                                 imag[index]))
                #                        index += 1
                #                        j += 1
                #                    while j < nstates_vib:
                #                        fn.write(template.format(i+start+1, j+1, 0., 0.))
                #                        j += 1
                #                start += nstates
                #        if print_stdout:
                #            time_taken = timedelta(seconds=round(time()-time_start, 0))
                #            print("Wrote {} lines to {} in {}".format(int(nelem),
                #                                                      filename.format(val),
                #                                                      str(time_taken)))
            if print_stdout:
                print('='*68)
        # write the vibronic energies to file
        if False:
            if print_stdout:
                print('='*68)
                print("Printing vibronic energies to file")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir, 0o755)
            start = time()
            filename = os.path.join(out_dir, 'energies.txt')
            template = "{:.9f}\n"
            if sparse:
                with open(filename, 'w') as fn:
                    fn.write('# {:d} (atomic units)\n'.format(2*nmodes*nstates))
                    # iterate over the normal modes
                    for fdx in range(nmodes):
                        # iterate over the plus and minus vibronic states around the
                        # electronic excitation
                        for idx, val in enumerate(['minus', 'plus']):
                            # iterate over all upper triangular elements
                            # this file does not need any fancy row or column indexing
                            for i in range(nstates):
                                fn.write(template.format(delta_E[fdx][idx][i]))
                    if print_stdout:
                        time_taken = timedelta(seconds=round(time()-start, 0))
                        print("Wrote {:d} lines to {} in {}".format(2*nmodes*nstates,
                                                                  filename, str(time_taken)))
            else:
                raise Exception("I will not write a non-sparse matrix....Too many zeros")
                #with open(filename, 'w') as fn:
                #    fn.write('# {} (atomic units)\n'.format(nelem))
                #    time_start = time()
                #    nelem = nstates_vib**2
                #    start = 0
                #    # iterate over the normal modes
                #    for fdx in range(nmodes):
                #        # iterate over the plus and minus vibronic states around the
                #        # electronic excitation
                #        for idx, val in enumerate(['minus', 'plus']):
                #            index = 0
                #            for i in range(nstates):
                #                j = 0
                #                while j < i+start:
                #                    fn.write(template.format(0))
                #                    j += 1
                #                while (j < start+nstates and j%nstates >= i):
                #                    fn.write(template.format(delta_E[fdx][idx][index]))
                #                    index += 1
                #                    j += 1
                #                while j < nstates_vib:
                #                    fn.write(template.format(0))
                #                    j += 1
                #            start += nstates
                #    if print_stdout:
                #        time_taken = timedelta(seconds=round(time()-time_start, 0))
                #        print("Wrote {} lines to {} in {}".format(nelem, filename, str(time_taken)))
            if print_stdout:
                print('='*68)
        # write the oscillators to file if available
        if write_oscil:
            if print_stdout:
                print('='*68)
                print("Writing vibronic oscillators to file")
            if not os.path.exists(out_dir):
                os.mkdir(out_dir, 0o755)
            start = time()
            filename = os.path.join(out_dir, 'oscillators.txt')
            with open(filename, 'w') as fn:
                # we also write the energies as they are needed when plotting the oscillators
                template = "{:6d}  {:6d}  {:+.16E}  {:+.16E}  {:4d}  {:7s}\n"
                fn.write('#NROW NCOL OSCIL ENERGY FREQDX SIGN\n')
                # outside of for loops as this will go from 0 to the number of vibronic states
                initial = 0
                # iterate over the normal modes
                for fdx, founddx in enumerate(found_modes):
                    # iterate over the plus and minus vibronic states around the
                    # electronic excitation
                    for idx, val in enumerate(['minus', 'none', 'plus']):
                        index = 0
                        # iterate over the rows
                        for i in range(nstates):
                            # this needs to reset to the initial value after we go to the
                            # next 'row' in the data
                            final = initial
                            # iterate over the columns
                            for j in range(nstates):
                                osc = oscil[fdx][idx][index]
                                energ = delta_E[fdx][idx][index]
                                #fn.write(template.format(initial+1, final+1, osc, energ, fdx, val))
                                fn.write(template.format(i+1, j+1, osc, energ, founddx, val))
                                index += 1
                                final += 1
                            initial += 1
                if print_stdout:
                    time_taken = timedelta(seconds=round(time()-start, 0))
                    print("Wrote {} lines to {} in {}".format(int(2*nselected*nstates*nstates),
                                                              filename, str(time_taken)))
            if print_stdout:
                print('='*68)
        program_end = time()
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
        config = Config.open_config(config_file, self._required_inputs, defaults=self._default_inputs)
        # check that the number of multiplicities and states are the same
        if len(config.spin_multiplicity) != len(config.number_of_states):
            print(config.spin_multiplicity, config.number_of_states)
            raise ValueError("Length mismatch of SPIN_MULTIPLICITY and NUMBER_OF_STATES")
        if len(config.spin_multiplicity) != int(config.number_of_multiplicity):
            print(config.spin_multiplicity, config.number_of_multiplicity)
            raise ValueError("Length of SPIN_MULTIPLICITY does not equal the NUMBER_OF_MULTIPLICITY")
        nstates = 0
        nstates_sf = 0
        for mult, state in zip(config.spin_multiplicity, config.number_of_states):
            nstates += mult*state
            nstates_sf += state
        self.config = config
        self.nstates = nstates
        self.nstates_sf = nstates_sf

