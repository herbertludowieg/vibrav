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
from exatomic.exa.util.units import Time, Length
from exatomic.util.constants import (speed_of_light_in_vacuum as speed_of_light,
                                Planck_constant as planck_constant)
from exatomic.util import conversions as conv
from vibrav.numerical.vibronic_func import (compute_oscil_str, compute_d_dq_sf,
                                            sf_to_so, compute_d_dq)
from vibrav.core.config import Config
from vibrav.numerical.degeneracy import energetic_degeneracy
from vibrav.numerical.boltzmann import boltz_dist
from vibrav.util.io import open_txt, write_txt
from vibrav.util.math import get_triu, ishermitian, isantihermitian, abs2
from vibrav.util.print import dataframe_to_txt
from vibrav.util.file_checking import _check_file_continuity
from vibrav.util.data_checking import check_size
from datetime import datetime, timedelta
from time import time

class Vibronic:
    '''
    Main class to run vibronic coupling calculations.

    **Required arguments in configuration file.**

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

    **Default arguments in configuration file.**

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

    **THEORY**

    Here, we present a brief account of the theory that is implemented in this
    module. The equations listed below are taken from *J. Phys. Chem. Lett.*
    **2018**, 9, 4, 887-994 (DOI: `10.1021/acs.jpclett.7b03441
    <https://doi.org/10.1021/acs.jpclett.7b03441>`_) and *J. Chem. Theory
    Comput.* **2020**, 16, 8, 5189-5202 (DOI: `10.1021/acs.jctc.0c00386
    <https://doi.org/10.1021/acs.jctc.0c00386>`_).

    The `vibronic` method in this class computes the Herzberg-Teller
    approximation in a via a Sum-over-states perturbation theory approach

    .. math::
        A = \\sum_{k\\neq 1}\\left<\\psi_k^0|\\Lambda^e|\\psi_2^0\\right>
            \\frac{\\partial\\left<\\psi_1^0|H|\\psi_k^0\\right>  / \\partial
            Q_p}{E_1^0 - E_k^0}

    .. math::
        B = \\sum_{k\\neq 2}\\left<\\psi_1^0|\\Lambda^e|\\psi_k^0\\right>
            \\frac{\\partial\\left<\\psi_1^0|H|\\psi_k^0\\right> / \\partial
            Q_p}{E_1^0 - E_k^0}

    .. math::
        \\frac{\\partial\\Lambda_{1,2}^{e}\\left(Q\\right)}{\\partial Q_p} = A+B

    Where, :math:`\\Lambda` can be the electric, magnetic, or quadrupolar
    transition moments between any two states, 1, and 2.  The equations above
    are computed by the
    :func:`vibrav.numerical.vibronic_func.compute_d_dq_sf` function.

    And spin-orbit coupling is then applied via

    .. math::
        \\left<\\psi_1^{SO}|\\mu^e|\\psi_2^{SO}\\right> =
            \\sum_{k,m}U_{k1}^{0*}U_{m2}^{0}
            \\left<\\psi_k|\\mu_{1,2}^{e,SF}\\left(Q\\right)|\\psi_m\\right>

    Where, :math:`U_{ij}^{0}` is an element of the complex tranformation matrix
    from a set of spin-free to spin-orbit coupled states, and
    :math:`\\left<\\psi_k|\\mu_{1,2}^{e,SF}\\left(Q\\right)|\\psi_m\\right>` is
    the vibronic transition moment for a pair of spin-free states. This is, in
    practice, the transition moment derivative calculated by
    :func:`vibrav.numerical.vibronic_func.compute_d_dq_sf`. The equation above
    is computed by the :func:`vibrav.numerical.vibronic_func:compute_d_dq`
    function.

    For an example on how to use this class please refer to the example that is
    available on the documentation under 'Examples'.
    '''
    _required_inputs = {'number_of_multiplicity': int,
                        'spin_multiplicity': (tuple, int),
                        'number_of_states': (tuple, int), 'number_of_nuclei': int,
                        'number_of_modes': int, 'zero_order_file': str}
    _skip_defaults = ['smatrix_file', 'eqcoord_file', 'atom_order_file']
    _default_inputs = {'sf_energies_file': ('', str), 'so_energies_file': ('', str),
                       'degen_delta': (1e-7, float),
                       'eigvectors_file': ('eigvectors.txt', str),
                       'so_cont_tol': (None, float), 'sparse_hamiltonian': (False, bool),
                       'states': (None, int), 'read_hamil': (False, bool),
                       'hamil_csv_file': (None, str)}
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

    @staticmethod
    def _init_oscil_file(fp):
        header = "{:>5s} {:>5s} {:>24s} {:>24s} {:>6s} {:>7s}".format
        oscil_formatters = ['{:>5d}'.format]*2+['{:>24.16E}'.format]*2 \
                           +['{:>6d}'.format, '{:>7s}'.format]
        for idx in [0,1,2,3]:
            with open(fp.format(idx), 'w') as fn:
                fn.write(header('#NROW', 'NCOL', 'OSCIL',
                                'ENERGY', 'FREQDX', 'SIGN'))

    @staticmethod
    def _write_prop_files(arr, dtemp, fname, fdx):
        for idx, arr in enumerate(zip(*arr)):
            for name in ['minus', 'plus']:
                dir_name = os.path.join(dtemp(fdx+1), name)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, 0o755, exist_ok=True)
                filename = os.path.join(dir_name, fname(idx+1))
                write_txt(arr, filename)

    def _parse_energies(self, ed, sf_file='', so_file=''):
        '''
        Parse the spin-free or spin-orbit energies from file. Will change
        depending on which file is given in the function call.

        Note:
            The files given in the `sf_file` and `so_file` inputs are assumed to
            have one column with no header.

        Args:
            ed (:class:`exatomic.exa.core.Editor`): Editor class that has
                already loaded the output file that contains the spin-free or
                spin-orbit energies. There is no default as this is only to be
                used in this class.
            sf_file (:obj:`str`, optional): Path to the file containing the
                spin-free energies.
            so_file (:obj:`str`, optional): Path to the file containing the
                spin-orbit energies.
        '''
        # parse the energies from the output is the energy files are not available
        if sf_file != '':
            try:
                energies_sf = pd.read_csv(self.config.sf_energies_file, header=None,
                                          comment='#').values.reshape(-1,)
            except FileNotFoundError:
                text = "The file {} was not found. Reading the spin-free " \
                       +"energies directly from the zero order output file {}."
                warnings.warn(text.format(self.config.sf_energies_file,
                                          config.zero_order_file), Warning)
                ed.parse_sf_energy()
                energies_sf = ed.sf_energy['energy'].values
        else:
            ed.parse_sf_energy()
            energies_sf = ed.sf_energy['energy'].values
        check_size(energies_sf, (self.nstates_sf,), 'energies_sf')
        if so_file != '':
            try:
                energies_so = pd.read_csv(self.config.so_energies_file, header=None,
                                          comment='#').values.reshape(-1,)
            except FileNotFoundError:
                text = "The file {} was not found. Reading the spin-orbit " \
                       +"energies directly from the zero order output file {}."
                warnings.warn(text.format(self.config.so_energies_file,
                                          config.zero_order_file), Warning)
                ed.parse_so_energy()
                energies_so = ed.so_energy['energy'].values
        else:
            ed.parse_so_energy()
            energies_so = ed.so_energy['energy'].values
        check_size(energies_so, (self.nstates,), 'energies_so')
        return energies_sf, energies_so

    def _write_oscil_file(self, fp_temp, boltz, arr, energies, evib, nstates,
                          fdx, ncomp, write_all_oscil, print_stdout):
        '''
        Method for writting the oscillators to file. This was created as the
        algorithm to write the spin-free and spin-orbit oscillators are exactly
        the same.

        Args:
            fp_temp (:obj:`str`): Filename template to use. We will place it
                into the 'vibronic-outputs' directory internally.
            boltz (:obj:`pandas.DataFrame`): Data frame with the Boltzmann
                distribution data.
            arr (:obj:`numpy.ndarray`): Array with the pertinent electric
                transition dipole moment data.
            energies (:obj:`numpy.ndarray`): Array with the pertinent energies
                in atomic units.
            evib (:obj:`float`): Vibrational energy in atomic units.
            nstates (:obj:`int`): Total number of states in the system. This
                will be used to test the sizes of the input arrays.
            fdx (:obj:`int`): Frequency index. Used only when writting the file.
            ncomp (:obj:`int`): Number of components in the given array. Should
                always be three as we only execute this when calculating the
                electric transition dipole moments.
            write_all_oscil (:obj:`bool`): Write all of the oscillators that are
                calculated. This include the unphysical negative values.
            print_stdout (:obj:`bool`): Print some text to the standard output.
        '''
        def to_file(fp, nr, nc, osc, en, fdx, sign, all_osc):
            template = ['{:>5d}']*2 + ['{:>24.16E}']*2 \
                                + ['{:>6d}', '{:>7s}']
            df = pd.DataFrame.from_dict({'nrow': nr, 'ncol': nc, 'oscil': osc,
                                         'energy': en})
            df['freqdx'] = fdx
            df['sign'] = sign
            if not all_osc:
                idxs = np.logical_and(df['oscil'].values > 0,
                                      df['energy'].values > 0)
                df = df.loc[idxs]
            write_txt(df, fp, non_matrix=True, mode='a', formatter=template)
        mapper = {0: 'iso', 1: 'x', 2: 'y', 3: 'z'}
        # finally get the oscillator strengths from equation S12
        nrow = np.tile(range(nstates), nstates)
        ncol = np.repeat(range(nstates), nstates)
        for idx, (val, sign) in enumerate(zip([-1, 1], ['minus', 'plus'])):
            boltz_factor = boltz.loc[fdx, sign]
            absorption = abs2(arr[idx].reshape(ncomp, nstates*nstates))
            # get the transition energies
            energy = energies.reshape(-1, 1) - energies.reshape(-1,) + val*evib
            energy = energy.flatten()
            # check for correct size
            check_size(energy, (nstates*nstates,), 'energy')
            check_size(absorption, (ncomp, nstates*nstates), 'absorption')
            # compute the isotropic oscillators
            oscil = boltz_factor * 2./3. * compute_oscil_str(np.sum(absorption, axis=0),
                                                             energy)
            # write to file
            filename = os.path.join('vibronic-outputs', fp_temp.format(0))
            start = time()
            to_file(filename, nrow, ncol, oscil, energy, fdx, sign,
                    write_all_oscil)
            if print_stdout:
                text = " Wrote isotropic oscillators to {} for sign {} in {:.2f} s"
                print(text.format(filename, sign, time() - start))
            # compute the oscillators for the individual cartesian components
            for idx, component in enumerate(absorption):
                check_size(component, (nstates*nstates,),
                           'absorption component {}'.format(idx))
                oscil = boltz_factor * 2. * compute_oscil_str(component, energy)
                filename = os.path.join('vibronic-outputs', fp_temp.format(idx+1))
                to_file(filename, nrow, ncol, oscil, energy, fdx, sign,
                        write_all_oscil)
                if print_stdout:
                    start = time()
                    text = " Wrote oscillators for {} component to {} for sign " \
                           +"{} in {:.2f} s"
                    print(text.format(mapper[idx+1], filename, sign, time() - start))

    def get_hamiltonian_deriv(self, delta, redmass, nmodes, select_fdx=-1,
                              use_sqrt_rmass=True, sparse_hamiltonian=False,
                              read_hamil=False, hamil_file=None):
        '''
        Find and read all of the Hamiltonian txt files in the different confg
        directories.

        Note:
            The path of confg is hardcoded along with the names of the SF
            Hamiltonian files as `'ham-sf.txt'`.

        Args:
            delta (:class:`pd.DataFrame`): Data frame with all the delta values
                    used for each of the normal mode displacements.  Read from
                    the given `delta_file` input in the configuration file.
            redmass (:class:`pd.DataFrame`): Data frame with all of the reduced
                    masses for each of the normal modes. Read from the given
                    `redmass_file` input in the configuration file.
            nmodes (int): The number of normal modes in the molecule.
            select_fdx (:obj:`list`, optional):
            use_sqrt_rmass (:obj:`bool`, optional): The calculations used
                    mass-weighted normal modes for the displaced structures.
                    This should always be the case. Defaults to `True`.
            sparse_hamiltonian (:obj:`bool`, optional): Input Hamiltonian files
                    are sparse matrices made up of block diagonal values.
                    Defaults to `False`.
            read_hamil (:obj:`bool`, optional): Read the Hamiltonians from a CSV
                    file. Defaults to `False`.
            hamil_file (:obj:`str`, optional): Path to the CSV file with the
                    Hamiltonian data. It expects that there will be an index
                    column and header row. In addition we expect there to be a
                    column called `freqdx` which is an index spanning 0 to
                    2*nmodes. Defaults to `None`.

        Returns:
            dham_dq (pd.DataFrame): Data frame with the derivative of the
                    Hamiltonians with respect to the normal mode.
        '''
        # read the hamiltonian files in each of the confg??? directories
        # it is assumed that the directories are named confg with a 3-fold padded number (000)
        plus_matrix = []
        minus_matrix = []
        found_modes = []
        dir_temp = 'confg{:03d}'.format
        if isinstance(select_fdx, (list, tuple, np.ndarray)):
            if select_fdx[0] == -1 and len(select_fdx) == 1:
                select_fdx = select_fdx[0]
            elif select_fdx[0] != -1:
                pass
            else:
                raise ValueError("The all condition for selecting frequencies " \
                                 +"(-1) was passed along with other frequencies.")
        if select_fdx == -1:
            freq_range = np.array(list(range(1, nmodes+1)))
        else:
            if isinstance(select_fdx, int): select_fdx = [select_fdx]
            freq_range = np.array(select_fdx) + 1
        nselected = len(freq_range)
        if read_hamil:
            for idx in freq_range:
                # error catching serves the purpose to know which
                # of the hamiltonian files are missing
                try:
                    plus = open_txt(os.path.join(dir_temp(idx), 'ham-sf.txt'),
                                    fill=sparse_hamiltonian)
                    try:
                        minus = open_txt(os.path.join(dir_temp(idx+nmodes), 'ham-sf.txt'),
                                         fill=sparse_hamiltonian)
                    except FileNotFoundError:
                        warnings.warn("Could not find ham-sf.txt file for in directory " \
                                      +dir_temp(idx+nmodes) \
                                      +". Ignoring frequency index {}".format(idx), Warning)
                        continue
                except FileNotFoundError:
                    warnings.warn("Could not find ham-sf.txt file for in directory " \
                                  +dir_temp(idx) \
                                  +". Ignoring frequency index {}".format(idx), Warning)
                    continue
                # put it all together only if both plus and minus
                # ham-sf.txt files are found
                plus['freqdx'] = idx
                minus['freqdx'] = idx+nmodes
                plus_matrix.append(plus)
                minus_matrix.append(minus)
                found_modes.append(idx-1)
            ham_plus = pd.concat(plus_matrix, ignore_index=True)
            ham_minus = pd.concat(minus_matrix, ignore_index=True)
        else:
            def filt_func(df, idxs):
                return df['freqdx'].unique() in idxs
            full_ham = pd.read_csv(hamil_file, header=0, index_col=0)
            if select_fdx != -1:
                full_ham = full_ham.groupby('freqdx') \
                            .filter(filt_func, idxs=list(freq_range) \
                                        + [x+nmodes for x in freq_range])
            full_ham = _check_file_continuity(full_ham, 'Hamiltonian', nmodes,
                                              col_name='freqdx')
            ham_plus = full_ham.groupby('freqdx') \
                            .filter(filt_func, idxs=range(1,nmodes+1)) \
                            .reset_index(drop=True)
            ham_minus = full_ham.groupby('freqdx') \
                            .filter(filt_func, idxs=range(nmodes+1,2*nmodes+1)) \
                            .reset_index(drop=True)
            found_modes = ham_plus['freqdx'].unique() - 1
            if len(found_modes) != len(freq_range):
                raise ValueError("Could not find all of the selected normal modes " \
                                 +"in the given Hamiltonian CSV file.")
        dham_dq = ham_plus - ham_minus
        dham_dq['freqdx'] = np.repeat(found_modes, dham_dq.shape[1]-1)
        data_cols = dham_dq.columns[dham_dq.columns != 'freqdx']
        if nselected != len(found_modes):
            warnings.warn("Number of selected normal modes is not equal to found modes, " \
                         +"currently, {} and {}\n".format(nselected, len(found_modes)) \
                         +"Overwriting the number of selceted normal modes by the number "\
                         +"of found modes.", Warning)
            nselected = len(found_modes)
        check_size(dham_dq, (self.nstates_sf*nselected, self.nstates_sf+1), 'dham_dq')
        # TODO: this division by the sqrt of the mass needs to be verified
        #       left as is for the time being as it was in the original code
        sf_sqrt_rmass = np.repeat(np.sqrt(redmass.loc[found_modes].values*(1/conv.amu2u)),
                                  self.nstates_sf).reshape(-1, 1)
        sf_delta = np.repeat(delta.loc[found_modes].values, self.nstates_sf).reshape(-1, 1)
        if use_sqrt_rmass:
            to_dq = 2 * sf_sqrt_rmass * sf_delta
        else:
            warnings.warn("We assume that you used non-mass-weighted " \
                          +"displacements to generate the displaced structures. " \
                          +"We cannot ensure that this actually works.",
                          Warning)
            to_dq = 2 * sf_delta
        # convert to normal coordinates
        dham_dq[data_cols] = dham_dq[data_cols] / to_dq
        # add a frequency index reference
        return dham_dq

    def vibronic_coupling(self, prop_name, write_property=True,
                          write_energy=True, write_oscil=True,
                          print_stdout=True, temp=298, eq_cont=False,
                          verbose=False, use_sqrt_rmass=True, select_fdx=-1,
                          boltz_states=None, boltz_tol=1e-6,
                          write_sf_oscil=False, write_sf_property=False,
                          write_dham_dq=False, write_all_oscil=False):
        '''
        Vibronic coupling method to calculate the vibronic coupling by the
        equations as given in reference *J. Phys. Chem. Lett.* **2018**, 9,
        887-894. This code follows a similar structure as that from a perl
        script written by Yonaton Heit and Jochen Autschbach, authors of the
        cited paper.

        Note:
            The script is able to calculate the vibronic contributions to the
            electric_dipole, magnetic_dipole and electric_quadrupole, currently.
            For more properties please reach out through github or via email.

            This will only work with data from Molcas/OpenMolcas.

        Warning:
            The value of the energy difference parameter (`degen_delta` in the
            configuration file) and the spin-free contribution to the spin-orbit
            states cutoff (`so_cont_tol` in the configuration file) can be very
            important in giving "reasonable" vibronic intensities. These values
            should be adjusted and tested accordingly on a per-system basis.
            **We make no guarantees everything will work out of the box**.

        Args:
            prop_name (:obj:`str`): Property of interest to calculate.
            write_property (:obj:`bool`, optional): Write the calculated
                    vibronic property values to file.  Defaults to `True`.
            write_energy (:obj:`bool`, optional): Write the vibronic energies to
                    file.  Defaults to `True`.
            write_oscil (:obj:`bool`, optional): Write the vibronic oscillators
                    to file.  Defaults to `True`.
            print_stdout (:obj:`bool`, optional): Print the progress of the
                    script to stdout.  Defaults to `True`.
            temp (:obj:`float`, optional): Temperature for the boltzmann
                    statistics. Defaults to 298.
            verbose (:obj:`bool`, optional): Send all availble print statements
                    listing where the program is in the calculation to stdout
                    and timings. Recommended if you have a system with many
                    spin-orbit states. Defaults to `False`.
            use_sqrt_rmass (:obj:`bool`, optional): The calculations used
                    mass-weighted normal modes for the displaced structures.
                    This should always be the case. Defaults to `True`.
            select_fdx (:obj:`list`, optional): Only use select normal modes in
                    the vibronic coupling calculation. Defaults to `-1` (all
                    normal modes).
            boltz_states (:obj:`int`, optional): Boltzmann states to calculate
                    in the distribution.  Defaults to `None` (all states with a
                    distribution less than the `boltz_tol` value for the lowest
                    frequency).
            boltz_tol (:obj:`float`, optional): Tolerance value for the
                    Boltzmann distribution cutoff.  Defaults to `1e-5`.
            write_sf_oscil (:obj:`bool`, optional): Write the spin-free vibronic
                    oscillators.  Defaults to `False`.
            write_sf_property (:obj:`bool`, optional): Write the spin-free
                    vibronic property values.  Defaults to `False`.
            write_dham_dq (:obj:`bool`, optional): Write the hamiltonian
                    derivatives for each normal mode. Defaults to `False`.
            write_all_oscil (:obj:`bool`, optional): Write the entire matrix of
                    the vibronic oscillator values instead of only those that
                    are physically meaningful (positive energy and oscillator
                    value).  Defaults to `False`.

        Raises:
            NotImplementedError: When the property requested with the `property`
                    parameter does not have any output parser or just has not
                    been coded yet.
            ValueError: If the array that is expected to be Hermitian actually
                    is not.
        '''
        # 90% of this method is actually just error checking and making
        # sure that the input data is what is to be expected
        # oftentimes this is actually the more important step as it is
        # possible to get a result with bad data and it is a bug that
        # will not be noticed
        # if the program throws an error there is a reason why that happens
        # if the error is not documented or thrown by something other than
        # this program feel free to contact the administrators on github
        # to look into the issue
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
        # deprecated because there are issues with the printing algorithm
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
        check_size(multiplicity, (nstates_sf,), 'multiplicity')
        # read the eigvectors data
        eigvectors = open_txt(config.eigvectors_file).values
        # mainly for testing purposes but this serves the purpose of limiting
        # the contribution of the SOC from states that can cause some issues
        # with the final intensities
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
        check_size(eigvectors, (nstates, nstates), 'eigvectors')
        # get the hamiltonian derivatives
        hamil_kwargs = dict(select_fdx=select_fdx, delta=delta, redmass=rmass,
                            nmodes=nmodes, use_sqrt_rmass=use_sqrt_rmass,
                            sparse_hamiltonian=config.sparse_hamiltonian,
                            read_hamil=config.read_hamil,
                            hamil_file=config.hamil_csv_file)
        dham_dq = self.get_hamiltonian_deriv(**hamil_kwargs)
        found_modes = dham_dq['freqdx'].unique()
        # TODO: it would be really cool if we could just input a list of properties to compute
        #       and the program will take care of the rest
        ed = Output(config.zero_order_file)
        # get the property of choice from the zero order file given in the config file
        # the extra column in each of the parsed properties comes from the component column
        # in the molcas output parser
        if prop_name.replace('_', '-') == 'electric-dipole':
            ed.parse_sf_dipole_moment()
            check_size(ed.sf_dipole_moment, (nstates_sf*3, nstates_sf+1), 'sf_dipole_moment')
            grouped_data = ed.sf_dipole_moment.groupby('component')
            out_file = 'dipole'
            #so_file = config.dipole_file
            idx_map = {1: 'x', 2: 'y', 3: 'z'}
        elif prop_name.replace('_', '-') == 'electric-quadrupole':
            ed.parse_sf_quadrupole_moment()
            check_size(ed.sf_quadrupole_moment, (nstates_sf*6, nstates_sf+1),
                             'sf_quadrupole_moment')
            grouped_data = ed.sf_quadrupole_moment.groupby('component')
            out_file = 'quadrupole'
            #so_file = config.quadrupole_file
            idx_map = {1: 'xx', 2: 'xy', 3: 'xz', 4: 'yy', 5: 'yz', 6: 'zz'}
        elif prop_name.replace('_', '-') == 'magnetic-dipole':
            ed.parse_sf_angmom()
            check_size(ed.sf_angmom, (nstates_sf*3, nstates_sf+1), 'sf_angmom')
            grouped_data = ed.sf_angmom.groupby('component')
            out_file = 'angmom'
            #so_file = config.angmom_file
            idx_map = {1: 'x', 2: 'y', 3: 'z'}
        else:
            raise NotImplementedError("Sorry the attribute that you are trying to use is not " \
                                     +"yet implemented.")
        # deprecated
        #if eq_cont:
        #    # get the spin-orbit property from the molcas output for the equilibrium geometry
        #    dfs = []
        #    for file in glob(so_file+'-?.txt'):
        #        idx = int(file.split('-')[-1].replace('.txt', ''))
        #        df = open_txt(file)
        #        # use a mapper as we cannot ensure that the files are found in any
        #        # expected order
        #        df['component'] = idx_map[idx]
        #        dfs.append(df)
        #    so_props = pd.concat(dfs, ignore_index=True)
        #
        # number of components
        # important because we can have up to 6 components for the
        # electric quadrupole moments
        ncomp = len(idx_map.keys())
        # for easier access
        idx_map_rev = {v: k for k, v in idx_map.items()}
        # get the energies
        energies_sf, energies_so = self._parse_energies(ed, config.sf_energies_file,
                                                        config.so_energies_file)
        # more testing things but this will only include a select number of the
        # sf states in the SOS equations
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
        if write_oscil and prop_name.replace('_', '-') == 'electric-dipole':
            osc_tmp = 'oscillators-{}.txt'
            self._init_oscil_file(os.path.join(vib_dir, osc_tmp))
        if write_sf_oscil and prop_name.replace('_', '-') == 'electric-dipole':
            osc_tmp = 'oscillators-sf-{}.txt'
            self._init_oscil_file(os.path.join(vib_dir, osc_tmp))
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
            check_size(dham_dq_mode, (nstates_sf, nstates_sf), 'dham_dq_mode')
            tdm_prefac = np.sqrt(planck_constant_au \
                                 /(2*speed_of_light_au*freq[founddx]/Length['cm', 'au']))/(2*np.pi)
            if print_stdout:
                print("TDM prefac: {:.4f}".format(tdm_prefac))
            prefactor.append(tdm_prefac)
            # iterate over all of the available components
            for idx, (key, val) in enumerate(grouped_data):
                start = time()
                # get the values of the specific component
                prop = val.drop('component', axis=1).values
                check_size(prop, (nstates_sf, nstates_sf), 'prop_{}'.format(key))
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
                if prop_name == 'electric_dipole':
                    if not ishermitian(dprop_dq):
                        text = "The vibronic electric dipole at frequency {} for component {} " \
                               +"was not found to be hermitian."
                        raise ValueError(text.format(fdx, key))
                    if not ishermitian(dprop_dq_sf):
                        text = "The vibronic electric dipole at frequency {} for component {} " \
                               +"was not found to be hermitian."
                        raise ValueError(text.format(fdx, key))
                elif prop_name == 'magnetic_dipole':
                    if not isantihermitian(dprop_dq):
                        text = "The vibronic magentic dipole at frequency {} for component {} " \
                               +"was not found to be non-hermitian."
                        raise ValueError(text.format(fdx, key))
                elif prop_name == 'electric_quadrupole':
                    if not ishermitian(dprop_dq):
                        text = "The vibronic electric quadrupole at frequency {} for " \
                               +"component {} was not found to be hermitian."
                        raise ValueError(text.format(fdx, key))
                dprop_dq *= tdm_prefac
                dprop_dq_sf *= tdm_prefac
                dprop_dq_so *= tdm_prefac
                # generate the full property vibronic states following equation S3 for the reference
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
            # TODO: This needs some revisions. Whole lot of spaghetti code.
            # no calculations from this point onward
            # just a whole lot of file writing
            if write_property:
                dtemp = 'vib{:03d}'.format
                fname = out_file+'-{}.txt'.format
                self._write_prop_files(vib_prop, dtemp, fname,fdx)
                signs = ['minus', 'plus']
                facs = [[1/2, 3/2], [3/2, 1/2]]
                for sign, fac in zip(signs, facs):
                    dir_name = os.path.join(dtemp(founddx+1), sign)
                    energies = energies_so + fac[0]*evib - energies_so[0]
                    energies[range(gs_degeneracy)] = energies_so[:gs_degeneracy] \
                                                        - energies_so[0] + fac[1]*evib
                    with open(os.path.join(dir_name, 'energies.txt'), 'w') as fn:
                        fn.write('# {} (atomic units)\n'.format(nstates))
                        for energy in energies:
                            fn.write('{:.9E}\n'.format(energy))
            if write_sf_property:
                dtemp = 'vib{:03d}'.format
                fname = out_file+'-sf-{}.txt'.format
                self._write_prop_files(vib_prop_sf, dtemp, fname, fdx)
            if write_sf_property:
                dtemp = 'vib{:03d}'.format
                fname = out_file+'-sf-so-len-{}.txt'.format
                self._write_prop_files(vib_prop_sf_so_len, dtemp, fname, fdx)
            if write_dham_dq:
                dtemp = 'vib{:03d}'.format
                fname = 'hamiltonian-derivs.txt'
                dir_name = dtemp(founddx+1)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name, 0o755)
                filename = os.path.join(dir_name, fname)
                write_txt(dham_dq_mode, filename)
            if (prop_name.replace('_', '-') == 'electric-dipole') and write_oscil:
                self._write_oscil_file('oscillators-{}.txt', boltz, vib_prop,
                                       energies_so, evib, nstates, founddx, 3,
                                       write_all_oscil, print_stdout)
            if (prop_name.replace('_', '-') == 'electric-dipole') and write_sf_oscil:
                self._write_oscil_file('oscillators-sf-{}.txt', boltz, vib_prop_sf,
                                       energies_sf, evib, nstates_sf, founddx, 3,
                                       write_all_oscil, print_stdout)
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
                                    defaults=self._default_inputs,
                                    skip_defaults=self._skip_defaults)
        # check that the number of multiplicities and states are the same
        if len(config.spin_multiplicity) != len(config.number_of_states):
            print(config.spin_multiplicity, config.number_of_states)
            msg = "Length mismatch of SPIN_MULTIPLICITY ({}) and " \
                  +"NUMBER_OF_STATES ({})"
            raise ValueError(msg.format(len(config.spin_multiplicity),
                                        len(config.number_of_states)))
        if len(config.spin_multiplicity) != int(config.number_of_multiplicity):
            print(config.spin_multiplicity, config.number_of_multiplicity)
            msg = "Length of SPIN_MULTIPLICITY ({}) does not equal the" \
                  +"NUMBER_OF_MULTIPLICITY ({})"
            raise ValueError(msg.format(config.spin_multiplicity,
                                        config.number_of_multiplicity))
        nstates = 0
        nstates_sf = 0
        for mult, state in zip(config.spin_multiplicity, config.number_of_states):
            nstates += mult*state
            nstates_sf += state
        self.config = config
        self.nstates = nstates
        self.nstates_sf = nstates_sf

