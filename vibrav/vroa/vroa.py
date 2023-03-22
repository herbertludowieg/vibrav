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
from exatomic.util import conversions, constants
from vibrav.numerical.vroa_func import backscat, forwscat, make_derivatives
from vibrav.core.config import Config
from vibrav.util.io import read_data_file
from vibrav.util.file_checking import _check_file_continuity
import numpy as np
import pandas as pd
import warnings

class VROA():
    '''
    Main class to run vibrational Raman optical activity calculations.

    Required arguments in the configuration file.

    +------------------------+--------------------------------------------------+----------------------------+
    | Argument               | Description                                      | Data Type                  |
    +========================+==================================================+============================+
    | number_of_nuclei       | Number of nuclei in the system.                  | :obj:`int`                 |
    +------------------------+--------------------------------------------------+----------------------------+
    | number_of_modes        | Number of normal modes in the molecule.          | :obj:`int`                 |
    +------------------------+--------------------------------------------------+----------------------------+
    | incident_frequency     | The incident frequency used in the calculation.  | :obj:`list` of             |
    |                        | This should have the same number of elements as  | :obj:`float`               |
    |                        | the unique labels in the `exc_idx` column.       |                            |
    |                        | Expected to be in units of nanometer.            |                            |
    +------------------------+--------------------------------------------------+----------------------------+

    Default arguments in configuration file specific to this class.

    +------------------+------------------------------------------------------------+----------------+
    | Argument         | Description                                                | Default Value  |
    +==================+============================================================+================+
    | roa_file         | Filepath of the ROA data from the quantum chemistry        | roa.csv        |
    |                  | calculation.                                               |                |
    +------------------+------------------------------------------------------------+----------------+
    | grad_file        | Filepath of the gradient data from the quantum chemistry   | grad.csv       |
    |                  | calculation.                                               |                |
    +------------------+------------------------------------------------------------+----------------+

    Other default arguments are taken care of with the :func:`vibrav.core.config.Config` class.

    '''
    _required_inputs = {'number_of_modes': int, 'number_of_nuclei': int,
                        'incident_frequency': (list, float)}
    _default_inputs = {'roa_file': ('roa.csv', str),
                       'grad_file': ('grad.csv', str)}
    @staticmethod
    def raman_int_units(lambda_0, lambda_p, temp=None):
        '''
        Function to calculate the K_p value as given in equation 2 on J. Chem.
        Phys. 2007, 127, 134101.
        We assume the temperature to be 298.15 as a hard coded value. Must get
        rid of this in future
        iterations. The final units of the equation are in cm^2/sr which are
        said to be the units for
        the Raman intensities.

        Note:
            Input values `lambda_0` and `lambda_p` must be in the units of
            m :math:`^{-1}`.

        Args:
            lambda_0 (float): Wavenumber value of the incident light
            lambda_1 (numpy.array): Wavenumber values of the vibrational modes
            temp (float): Value of the temperature of the experiment

        Returns:
            kp (numpy.array): Array with the values of the conversion units of
                              length lambda_1.shape[0]
        '''
        if temp is None: temp=298.15
        H = constants.Planck_constant
        C = constants.speed_of_light_in_vacuum
        KB = constants.Boltzmann_constant
        au2m = constants.atomic_unit_of_length
        u2Kg = conversions.u2Kg
        boltz = 1.0/(1.0-np.exp(-H*C*lambda_p/(KB*temp)))
        const = H * np.pi**2 / C
        variables = (lambda_0 - lambda_p)**4/lambda_p
        kp = variables * const * boltz * (au2m**4 / u2Kg) * 16 / 45. * 100**2
        return kp

    @staticmethod
    def make_complex(df):
        '''
        Transform the electric dipole-quadrupole polarizability tensor
        to complex valued.

        Args:
            df (pandas.DataFrame): Data frame with the three cartesian
                                directions of the electric
                                dipole-quadrupole polarizability tensor.

        Returns:
            new_df (pandas.DataFrame): Data frame with the complex
                                valued tensor.
        '''
        grouped = df.groupby('type')
        cols = [x+y for x in ['x', 'y', 'z'] for y in ['x', 'y', 'z']]
        complex_val = grouped.get_group('real')[cols].values \
                  + 1j*grouped.get_group('imag')[cols].values
        new_df = pd.DataFrame(complex_val, columns=cols)
        #new_df['file'] = df['file'].unique()[0]
        new_df['exc_idx'] = df['exc_idx'].unique()[0]
        return new_df

    def vroa(self, atomic_units=True, temp=None, assume_real=False, print_stdout=False):
        '''
        VROA method to calculate the VROA back/forwardscatter intensities from the
        equations given in paper J. Phys. Chem. A 2016, 120, 9740-9748
        DOI: 10.1021/acs.jpca.6b09975

        Note:
            The final units of this method is in :math:`\\unicode{xC5}^{4} / amu`. When using
            `atomic_units=False` the output values are in :math:`cm^2 / sr`.

        Args:
            atomic_units (:obj:`bool`, optional): Calculate the intensities in
                                    atomic units. Defaults to `True`.
            temp (:obj:`float`, optional): Calculate the boltzmann factors with
                                    the specified temperature. Defaults to
                                    `None` which is then converted to 298 K.
            assume_real (:obj:`bool`, optional): Assume that the ROA data is
                                    not complex valued. The equations will
                                    ignore the imaginary contributions. Only
                                    recommended for testing purposes.
                                    Defaults to `False`.
            print_stdout (:obj:`bool`, optional): Print the progress of the
                                    script to stdout. Defaults to `False`.
        '''
        config = self.config
        if print_stdout:
            print("Printing contents of config file")
            print("*"*46)
            print(config.to_string())
            print("*"*46)
        scatter = []
        raman = []
        # grab the data from the respective files given in the config file
        nmodes = config.number_of_modes
        nat = config.number_of_nuclei
        delta = read_data_file(config.delta_file, nmodes)
        rmass = read_data_file(config.reduced_mass_file, nmodes)
        freq = read_data_file(config.frequency_file, nmodes)
        # grab the data that was already parsed for the ROA and gradients
        roa = pd.read_csv(config.roa_file)
        grad = pd.read_csv(config.grad_file)
        try:
            # remove the zeroth index
            roa_0 = roa.groupby('file').get_group(0)
            idxs = roa_0.index.values
            roa = roa.loc[~roa.index.isin(idxs)]
        except KeyError:
            pass
        # set up some constants
        C = constants.speed_of_light_in_vacuum
        conv = constants.atomic_unit_of_time / constants.atomic_unit_of_length
        C_au = C * conv
        epsilon = np.array([[0,0,0,0,0,1,0,-1,0],
                            [0,0,-1,0,0,0,1,0,0],
                            [0,1,0,-1,0,0,0,0,0]])
        arr = zip(roa.groupby('exc_idx'), grad.groupby('exc_idx'))
        for _, ((idx, roa_data), (_, grad_data)) in enumerate(arr):
            # convert the excitation frequency to a.u.
            try:
                #tmp = roa_data['exc_freq'].unique()
                #if tmp.shape[0] > 1:
                #    raise ValueError("More than one excitation frequency was found " \
                #                     +"with the same index.")
                exc_wave = config.incident_frequency[idx]
                if exc_wave > 3000:
                    msg = "Detected very long wavelength values for " \
                          +"the incident frequency {} nm. Will " \
                          +"continue with the calculation with this " \
                          +"value."
                    warnings.warn(msg.format(exc_wave), Warning)
                if print_stdout:
                    print("Found excitation wavelength of {:.2f} nm".format(exc_wave))
                exc_freq = 1e9/exc_wave*conversions.inv_m2Ha
            except ZeroDivisionError:
                text = "The excitation frequency detected was close to zero"
                raise ZeroDivisionError(text)
            # check to see that we have a positive and negative displacement
            # for each of the normal modes
            roa_data = _check_file_continuity(roa_data, "ROA", nmodes)
            grad_data = _check_file_continuity(grad_data, "Gradient", nmodes)
            # get which of the frequencies we have
            select_freq = roa_data['file'].sort_values().drop_duplicates().values-1
            mask = select_freq > nmodes-1
            select_freq = select_freq[~mask]
            snmodes = len(select_freq)
            # get the data for those frequencies that we have found
            if snmodes < nmodes:
                sel_rmass = rmass[select_freq].reshape(snmodes,1)
                sel_delta = delta[select_freq].reshape(snmodes,1)
                sel_freq = freq[select_freq]
            else:
                sel_rmass = rmass.reshape(snmodes, 1)
                sel_delta = delta.reshape(snmodes, 1)
                sel_freq = freq
            cols = ['label', 'file']
            complex_roa = roa_data.groupby(cols).apply(self.make_complex)
            complex_roa.reset_index(inplace=True)
            complex_roa.drop('level_2', axis=1, inplace=True)
            # get the data for the dipole-quadrupole polarizability
            cols = [x+y for x in ['x', 'y', 'z'] for y in ['x', 'y', 'z']]
            # get the indeces of where those values are in the ROA dataframe
            # there are three cartesian directions as it is a 3D tensor
            index = list(map(lambda x: x == 'Ax', complex_roa['label']))
            index = np.logical_or(list(map(lambda x: x == 'Ay',
                                           complex_roa['label'])), index)
            index = np.logical_or(list(map(lambda x: x == 'Az',
                                           complex_roa['label'])), index)
            grouped = complex_roa.loc[index].groupby('file')
            tmp = grouped.apply(lambda x: np.array(
                                    [x[cols].values[0], x[cols].values[1],
                                     x[cols].values[2]]).flatten())
            tmp = tmp.reset_index(drop=True).to_dict()
            A = pd.DataFrame.from_dict(tmp).T
            # get the data for the electric dipole-dipole polarizability
            index = list(map(lambda x: x == 'alpha', complex_roa['label']))
            tmp = complex_roa.loc[index, cols].reset_index(drop=True).to_dict()
            alpha = pd.DataFrame.from_dict(tmp)
            # get the data for the electric dipole-magnetic dipole polarizability
            index = list(map(lambda x: x == 'g_prime', complex_roa['label']))
            tmp = complex_roa.loc[index, cols].reset_index(drop=True).to_dict()
            g_prime = pd.DataFrame.from_dict(tmp)
            # determine the derivatives of the gradients
            #grad_derivs = self.get_pos_neg_gradients(grad_data, smat, nmodes)
            # separate tensors into positive and negative displacements
            # highly dependent on the value of the index
            # we neglect the equilibrium coordinates
            # 0 corresponds to equilibrium coordinates
            # 1 - nmodes corresponds to positive displacements
            # nmodes+1 - 2*nmodes corresponds to negative displacements
            alpha_plus = np.divide(alpha.loc[range(0,snmodes)].values, np.sqrt(sel_rmass))
            alpha_minus = np.divide(alpha.loc[range(snmodes, 2*snmodes)].values, np.sqrt(sel_rmass))
            g_prime_plus = np.divide(g_prime.loc[range(0,snmodes)].values, np.sqrt(sel_rmass))
            g_prime_minus = np.divide(g_prime.loc[range(snmodes, 2*snmodes)].values, np.sqrt(sel_rmass))
            A_plus = np.divide(A.loc[range(0, snmodes)].values, np.sqrt(sel_rmass))
            A_minus = np.divide(A.loc[range(snmodes, 2*snmodes)].values, np.sqrt(sel_rmass))
            # generate derivatives by two point central finite difference method
            dalpha_dq = np.divide((alpha_plus - alpha_minus), 2 * sel_delta)
            dg_dq = np.divide((g_prime_plus - g_prime_minus), 2 * sel_delta)
            dA_dq = np.array([np.divide((A_plus[i] - A_minus[i]), 2 * sel_delta[i])
                                                                    for i in range(snmodes)])
            # generate properties as shown on equations 5-9 in paper
            # J. Chem. Phys. 2007, 127, 134101
            au2angs = constants.atomic_unit_of_length*1e10
            alpha_squared, beta_alpha, beta_g, beta_A, alpha_g = make_derivatives(dalpha_dq,
                                  dg_dq, dA_dq, exc_freq, epsilon, snmodes, au2angs**4, C_au, assume_real)
            # calculate Raman intensities
            raman_int = 4 * (45 * alpha_squared + 8 * beta_alpha)
            # calculate VROA back scattering and forward scattering intensities
            backscat_vroa = backscat(beta_g, beta_A)
            forwscat_vroa = forwscat(alpha_g, beta_g, beta_A)
            # convert to raman units if desired
            if not atomic_units:
                lambda_0 = exc_freq*conversions.Ha2inv_m
                lambda_p = sel_freq*100
                kp = self.raman_int_units(lambda_0=lambda_0, lambda_p=lambda_p, temp=temp)*100**2
                raman_int *= kp
                backscat_vroa *= kp
                forwscat_vroa *= kp
            # generate dataframe with all pertinent data for vroa scatter
            df = pd.DataFrame.from_dict({"freq": sel_freq, "freqdx": select_freq, "beta_g*1e6":beta_g*1e6,
                                        "beta_A*1e6": beta_A*1e6, "alpha_g*1e6": alpha_g*1e6,
                                        "backscatter": backscat_vroa, "forwardscatter":forwscat_vroa})
            df['exc_freq'] = np.repeat(exc_wave, len(df))
            df['exc_idx'] = np.repeat(idx, len(df))
            rdf = pd.DataFrame.from_dict({"freq": sel_freq, "freqdx": select_freq,
                                          "alpha_squared": alpha_squared,
                                          "beta_alpha": beta_alpha, "raman_int": raman_int})
            rdf['exc_freq'] = np.repeat(exc_wave, len(rdf))
            rdf['exc_idx'] = np.repeat(idx, len(df))
            scatter.append(df)
            raman.append(rdf)
        self.scatter = pd.concat(scatter)
        self.scatter.sort_values(by=['exc_freq','freq'], inplace=True)
        # added this as there seems to be some issues with the indexing when there are
        # nearly degenerate modes
        self.scatter.reset_index(drop=True, inplace=True)
        # check ordering of the freqdx column
        self.raman = pd.concat(raman)
        self.raman.sort_values(by=['exc_freq', 'freq'], inplace=True)
        self.scatter.reset_index(drop=True, inplace=True)

    def __init__(self, config_file, *args, **kwargs):
        config = Config.open_config(config_file, self._required_inputs,
                                    defaults=self._default_inputs)
        self.config = config

