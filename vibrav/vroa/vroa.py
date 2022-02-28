import numpy as np
import pandas as pd
import warnings
#from exa.util.constants import (speed_of_light_in_vacuum as C,
#                                Planck_constant as H,
#                                Boltzmann_constant as KB)
from exa.util import conversions, constants
from vibrav.numerical.vroa_func import _backscat, _forwscat, _make_derivatives
from vibrav.core.config import Config

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
    _required_inputs = {'number_of_modes': int, 'number_of_nuclei': int}
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
            Input values lambda_0 and lambda_p must be in the units of m^-1

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
    def _check_file_continuity(df, prop, nmodes):
        files = df['file'].drop_duplicates()
        pos_file = files[files.isin(range(1,nmodes+1))]
        neg_file = files[files.isin(range(nmodes+1, 2*nmodes+1))]-nmodes
        intersect = np.intersect1d(pos_file.values, neg_file.values)
        diff = np.unique(np.concatenate((np.setdiff1d(pos_file.values, intersect),
                                         np.setdiff1d(neg_file.values, intersect)), axis=None))
        rdf = df.copy()
        if len(diff) > 0:
            print("Seems that we are missing one of the {} outputs for frequency {} ".format(prop, diff)+ \
                  "we will ignore the {} data for these frequencies.".format(prop))
            rdf = rdf[~rdf['file'].isin(diff)]
            rdf = rdf[~rdf['file'].isin(diff+nmodes)]
        return rdf

    @staticmethod
    def get_pos_neg_gradients(grad, freq, nmodes):
        '''
        Here we get the gradients of the equilibrium, positive and negative displaced structures.
        We extract them from the gradient dataframe and convert them into normal coordinates
        by multiplying them by the frequency normal mode displacement values.

        Args:
            grad (:class:`exatomic.gradient.Gradient`): DataFrame containing all of the gradient data
            freq (:class:`exatomic.atom.Frquency`): DataFrame containing all of the frequency data

        Returns:
            delfq_zero (pandas.DataFrame): Normal mode converted gradients of equilibrium structure
            delfq_plus (pandas.DataFrame): Normal mode converted gradients of positive displaced structure
            delfq_minus (pandas.DataFrame): Normal mode converted gradients of negative displaced structure
        '''
        grouped = grad.groupby('file')
        # generate delta dataframe
        # TODO: make something so delta can be set
        #       possible issues are a user using a different type of delta
        #nmodes = len(freq['freqdx'].drop_duplicates().values)
        # get gradient of the equilibrium coordinates
        grad_0 = grouped.get_group(0)
        # get gradients of the displaced coordinates in the positive direction
        grad_plus = grouped.filter(lambda x: x['file'].drop_duplicates().values in
                                                                        range(1,nmodes+1))
        snmodes = len(grad_plus['file'].drop_duplicates().values)
        # get gradients of the displaced coordinates in the negative direction
        grad_minus = grouped.filter(lambda x: x['file'].drop_duplicates().values in
                                                                        range(nmodes+1, 2*nmodes+1))
        # TODO: Check if we can make use of numba to speed up this code
        delfq_zero = freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda x:
                                    np.sum(np.multiply(grad_0[['fx', 'fy', 'fz']].values, x.values))).values
        # we extend the size of this 1d array as we will perform some matrix summations with the
        # other outputs from this method
        delfq_zero = np.tile(delfq_zero, snmodes).reshape(snmodes, nmodes)

        delfq_plus = grad_plus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values)))).values
        delfq_minus = grad_minus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values)))).values
        return [delfq_zero, delfq_plus, delfq_minus]


    @staticmethod
    def make_complex(df):
        grouped = df.groupby('type')
        cols = [x+y for x in ['x', 'y', 'z'] for y in ['x', 'y', 'z']]
        complex = grouped.get_group('real')[cols].values \
                  + 1j*grouped.get_group('imag')[cols].values
        new_df = pd.DataFrame(complex, columns=cols)
        #new_df['file'] = df['file'].unique()[0]
        new_df['exc_freq'] = df['exc_freq'].unique()[0]
        new_df['exc_idx'] = df['exc_idx'].unique()[0]
        return new_df

    def vroa(self, atomic_units=True, temp=None, assume_real=False, print_stdout=False):
        '''
        VROA method to calculate the VROA back/forwardscatter intensities from the
        equations given in paper **insert paper**.

        Note:
            The final units of this method is in Angstrom^4 / amu. When using
            `atomic_units=False` the output values are in cm^2 / sr.

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
                                    Defaults to `False`raman_units.
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
        delta = pd.read_csv(config.delta_file,
                            header=None).values.reshape(-1)
        rmass = pd.read_csv(config.reduced_mass_file,
                            header=None).values.reshape(-1)
        freq = pd.read_csv(config.frequency_file,
                           header=None).values.reshape(-1)
        nmodes = config.number_of_modes
        nat = config.number_of_nuclei
        smat = pd.read_csv(config.smatrix_file, header=None)
        smat['groups'] = np.tile([0,1,2], nmodes*nat)
        tmp = smat.groupby('groups').apply(lambda x: x[0].values).to_dict()
        smat = pd.DataFrame.from_dict(tmp)
        smat.columns = ['dx', 'dy', 'dz']
        smat['freqdx'] = np.repeat(range(nmodes), nat)
        roa = pd.read_csv(config.roa_file)
        grad = pd.read_csv(config.grad_file)
        try:
            roa_0 = roa.groupby('file').get_group(0)
            idxs = roa_0.index.values
            roa = roa.loc[~roa.index.isin(idxs)]
        except KeyError:
            pass
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
                tmp = roa_data['exc_freq'].unique()
                if tmp.shape[0] > 1:
                    raise ValueError("More than one excitation frequency was found " \
                                     +"with the same index.")
                exc_wave = tmp[0]
                if print_stdout:
                    print("Found excitation wavelength of {:.2f} nm".format(exc_wave))
                exc_freq = 1e9/tmp[0]*conversions.inv_m2Ha
            except ZeroDivisionError:
                text = "The excitation frequency detected was close to zero"
                raise ZeroDivisionError(text)
            roa_data = self._check_file_continuity(roa_data, "ROA", nmodes)
            grad_data = self._check_file_continuity(grad_data, "Gradient", nmodes)
            select_freq = roa_data['file'].sort_values().drop_duplicates().values-1
            mask = select_freq > nmodes-1
            select_freq = select_freq[~mask]
            snmodes = len(select_freq)
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
            cols = [x+y for x in ['x', 'y', 'z'] for y in ['x', 'y', 'z']]
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
            index = list(map(lambda x: x == 'alpha', complex_roa['label']))
            tmp = complex_roa.loc[index, cols].reset_index(drop=True).to_dict()
            alpha = pd.DataFrame.from_dict(tmp)
            index = list(map(lambda x: x == 'g_prime', complex_roa['label']))
            tmp = complex_roa.loc[index, cols].reset_index(drop=True).to_dict()
            g_prime = pd.DataFrame.from_dict(tmp)
            grad_derivs = self.get_pos_neg_gradients(grad_data, smat, nmodes)
            
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

            # generate derivatives by two point difference method
            dalpha_dq = np.divide((alpha_plus - alpha_minus), 2 * sel_delta)
            dg_dq = np.divide((g_prime_plus - g_prime_minus), 2 * sel_delta)
            dA_dq = np.array([np.divide((A_plus[i] - A_minus[i]), 2 * sel_delta[i])
                                                                    for i in range(snmodes)])
            # generate properties as shown on equations 5-9 in paper
            # J. Chem. Phys. 2007, 127, 134101
            au2angs = constants.atomic_unit_of_length*1e10
            alpha_squared, beta_alpha, beta_g, beta_A, alpha_g = _make_derivatives(dalpha_dq,
                                  dg_dq, dA_dq, exc_freq, epsilon, snmodes, au2angs**4, C_au, assume_real)
            # calculate Raman intensities
            raman_int = 4 * (45 * alpha_squared + 8 * beta_alpha)

            # calculate VROA back scattering and forward scattering intensities
            backscat_vroa = _backscat(beta_g, beta_A)
            #backscat_vroa *= 1e4
            # TODO: check the units of this because we convert the invariants from
            #       au to Angstrom and here we convert again from au to Angstrom
            #backscat_vroa *= Length['au', 'Angstrom']**4*Mass['u', 'au_mass']
            #backscat_vroa *= Mass['u', 'au_mass']
            forwscat_vroa = _forwscat(alpha_g, beta_g, beta_A)
            #forwscat_vroa *= 1e4
            if not atomic_units:
                lambda_0 = exc_freq*conversions.Ha2inv_m
                lambda_p = sel_freq*100
                kp = self.raman_int_units(lambda_0=lambda_0, lambda_p=lambda_p, temp=temp)*100**2
                raman_int *= kp
                backscat_vroa *= kp
                forwscat_vroa *= kp
            # TODO: check the units of this because we convert the invariants from
            #       au to Angstrom and here we convert again from au to Angstrom
            #forwscat_vroa *=Length['au', 'Angstrom']**4*Mass['u', 'au_mass']
            # we set this just so it is easier to view the data
            pd.options.display.float_format = '{:.6f}'.format
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

