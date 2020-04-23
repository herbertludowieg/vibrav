from vibrav.core import Config
from exa.util.units import Length, Mass, Energy
from exa.util.constants import Boltzmann_constant as boltzmann
from exatomic.core.atom import Atom
from exatomic.base import sym2z
import numpy as np
import pandas as pd

class ZPVC:
    _required_inputs = {'number_of_modes': int, 'number_of_nuclei': int}
    _default_inputs = {'smatrix_file': ('smatrix.dat', str), 'eqcoord_file': ('eqcoord.dat', str),
                       'atom_order_file': ('atom_order.dat', str)}

    @staticmethod
    def _get_temp_factor(temp, freq):
        if temp > 1e-6:
            try:
                factor = freq*Energy['Ha', 'J'] / (2 * boltzmann * temp)
                temp_fac = np.cosh(factor) / np.sinh(factor)
            # this should be taken care of by the conditional but always good to
            # take care of explicitly
            except ZeroDivisionError:
                raise ZeroDivisionError("Something seems to have gone wrong with the sinh function")
        else:
            temp_fac = 1.
        return temp_fac

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
        #nmodes = len(smat)
        # get gradient of the equilibrium coordinates
        grad_0 = grouped.get_group(0)
        print(grad_0.shape)
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
        print(delfq_zero.shape, snmodes)
        delfq_zero = np.tile(delfq_zero, snmodes).reshape(snmodes, nmodes)

        delfq_plus = grad_plus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values)))).values
        delfq_minus = grad_minus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values)))).values
        return [delfq_zero, delfq_plus, delfq_minus]

    @staticmethod
    def calculate_frequencies(delfq_0, delfq_plus, delfq_minus, redmass, select_freq, delta=None):
        '''
        Here we calculated the frequencies from the gradients calculated for each of the
        displaced structures along the normal mode. In principle this should give the same or
        nearly the same frequency value as that from a frequency calculation.

        Args:
            delfq_0 (numpy.ndarray): Array that holds all of the information about the gradient
                                     derivative of the equlilibrium coordinates
            delfq_plus (numpy.ndarray): Array that holds all of the information about the gradient
                                        derivative of the positive displaced coordinates
            delfq_minus (numpy.ndarray): Array that holds all of the information about the gradient
                                         derivative of the negative displaced coordinates
            redmass (numpy.ndarray): Array that holds all of the reduced masses. We can handle both
                                     a subset of the entire values or all of the values
            select_freq (numpy.ndarray): Array that holds the selected frequency indexes
            delta (numpy.ndarray): Array that has the delta values used in the displaced structures

        Returns:
            frequencies (numpy.ndarray): Frequency array from the calculation
        '''
        if delta is None:
            print("No delta has been given. Assume delta_type to be 2.")
            delta = va.gen_delta(delta_type=2, freq=freq.copy())['delta'].values
        # get number of selected normal modes
        # TODO: check stability of using this parameter
        snmodes = len(select_freq)
        #print("select_freq.shape: {}".format(select_freq.shape))
        if len(redmass) > snmodes:
            redmass_sel = redmass[select_freq]
        else:
            redmass_sel = redmass
        if len(delta) > snmodes:
            delta_sel = delta[select_freq]
        else:
            delta_sel = delta
        # calculate force constants
        kqi = np.zeros(len(select_freq))
        #print(redmass_sel.shape)
        for fdx, sval in enumerate(select_freq):
            kqi[fdx] = (delfq_plus[fdx][sval] - delfq_minus[fdx][sval]) / (2.0*delta_sel[fdx])

        vqi = np.divide(kqi, redmass_sel.reshape(snmodes,))
        # TODO: Check if we want to exit the program if we get a negative force constant
        n_force_warn = vqi[vqi < 0.]
        if n_force_warn.any() == True:
            # TODO: point to exactly which frequencies are negative
            negative = np.where(vqi<0)[0]
            text = ''
            # frequencies are base 0
            for n in negative[:-1]: text += str(n)+', '
            text += str(negative[-1])
            warnings.warn("Negative force constants have been calculated for frequencies " \
                          +"{} be wary of results".format(text),
                          Warning)
        # return calculated frequencies
        frequencies = np.sqrt(vqi).reshape(snmodes,)*Energy['Ha', 'cm^-1']
        return frequencies

    def zpvc(self, gradient, property, temperature=None, geometry=True, print_results=False):
        """
        Method to compute the Zero-Point Vibrational Corrections. We implement the equations as
        outlined in the paper J. Phys. Chem. A 2005, 109, 8617-8623 (doi:10.1021/jp051685y).
        Here we compute the effect of vibrations on a specified property given as a n x 2 array
        where one of the columns are the file indexes and the other is the property.
        We use a two and three point difference method to calculate the first and second derivatives
        respectively.

        We have also implemented a way to calculate the ZPVC and effective geometries at
        different temperatures given in Kelvin.

        Note:
            The code has been designed such that the property input array must have one column
            labeled file corresponding to the file indexes.

        Args:
            uni (:class:`exatomic.Universe`): Universe containing all pertinent data
            delta (numpy.array): Array of the delta displacement parameters
            temperature (list): List object containing all of the temperatures of interest
            geometry (bool): Bool value that tells the program to also calculate the effective geometry
            print_results(bool): Bool value to print the results from the zpvc calcualtion to stdout
        """
        config = self.config
        if property.shape[1] != 2:
            raise ValueError("Property dataframe must have a second dimension of 2 not " \
                             +"{}".format(self.property.shape[1]))
        if temperature is None: temperature = [0]
        # get the total number of normal modes
        nmodes = config.number_of_modes
        # check for any missing files and remove the respective counterpart
        grad = self._check_file_continuity(gradient, 'gradient', nmodes)
        prop = self._check_file_continuity(property, 'property', nmodes)
        # check that the equlibrium coordinates are included
        # these are required for the three point difference methods
        try:
            tmp = grad.groupby('file').get_group(0)
        except KeyError:
            raise KeyError("Equilibrium coordinate gradients not found")
        try:
            tmp = prop.groupby('file').get_group(0)
        except KeyError:
            raise KeyError("Equilibrium coordinate property not found")
        # check that the gradient and property dataframe have the same length of data
        grad_files = grad[grad['file'].isin(range(0,nmodes+1))]['file'].drop_duplicates()
        prop_files = prop[prop['file'].isin(range(nmodes+1,2*nmodes+1))]['file'].drop_duplicates()
        # compare lengths
        # TODO: make sure the minus 1 is in the right place
        #       we suppose that it is because we grab the file number 0 as an extra
        if grad_files.shape[0]-1 != prop_files.shape[0]:
            print("Length mismatch of gradient and property arrays.")
            # we create a dataframe to make use of the existing file continuity checker
            df = pd.DataFrame(np.concatenate([grad_files, prop_files]), columns=['file'])
            df = self._check_file_continuity(df, 'grad/prop', nmodes)
            # overwrite the property and gradient dataframes
            grad = grad[grad['file'].isin(df['file'])]
            prop = prop[prop['file'].isin(df['file'])]
        # get the selected frequencies
        select_freq = grad[grad['file'].isin(range(1,nmodes+1))]
        select_freq = select_freq['file'].drop_duplicates().values - 1
        snmodes = len(select_freq)
        # get the actual frequencies
        # TODO: check if we should use the real or calculated frequencies
        frequencies = pd.read_csv(config.frequency_file, header=None).values.reshape(-1,)
        frequencies *= Energy['cm^-1','Ha']
        rmass = pd.read_csv(config.reduced_mass_file, header=None).values.reshape(-1,)
        rmass *= Mass['u', 'au_mass']
        delta = pd.read_csv(config.delta_file, header=None).values.reshape(-1,)
        eqcoord = pd.read_csv(config.eqcoord_file, header=None).values.reshape(-1,)
        nat = int(eqcoord.shape[0]/3)
        if nat != eqcoord.shape[0]/3.:
            raise ValueError("Something is wrong with the eqcoord file.")
        eqcoord = eqcoord.reshape(nat, 3)
        atom_symbols = pd.read_csv(config.atom_order_file, header=None).values.reshape(-1,)
        eqcoord = pd.DataFrame(eqcoord, columns=['x', 'y', 'z'])
        eqcoord['symbol'] = atom_symbols
        eqcoord['frame'] = 0
        eqcoord = Atom(eqcoord)
        smat = pd.read_csv(config.smatrix_file, header=None).values
        smat = smat.reshape(nmodes*nat, 3)
        smat = pd.DataFrame.from_dict(smat)
        smat.columns = ['dx', 'dy', 'dz']
        smat['freqdx'] = np.repeat(range(nmodes), nat)
        print(smat.to_string())
        print(grad.shape, smat.shape)
        # get the gradients multiplied by the normal modes
        delfq_zero, delfq_plus, delfq_minus = self.get_pos_neg_gradients(grad, smat, nmodes)
        if snmodes < nmodes:
            raise NotImplementedError("We do not currently have support to handle missing frequencies")
            #sel_delta = delta[select_freq]
            #sel_rmass = uni.frequency_ext['r_mass'].values[select_freq]*Mass['u', 'au_mass']
            #sel_freq = uni.frequency_ext['freq'].values[select_freq]*Energy['cm^-1','Ha']
        else:
            sel_delta = delta
            sel_rmass = rmass
            sel_freq = frequencies
        _ = self.calculate_frequencies(delfq_zero, delfq_plus, delfq_minus, sel_rmass, select_freq,
                                       sel_delta)
        # calculate cubic force constant
        # we use a for loop because we need the diagonal values
        # if we select a specific number of modes then the diagonal elements
        # are tricky
        kqiii = np.zeros(len(select_freq))
        for fdx, sval in enumerate(select_freq):
            kqiii[fdx] = (delfq_plus[fdx][sval] - 2.0 * delfq_zero[fdx][sval] + \
                                                delfq_minus[fdx][sval]) / (sel_delta[fdx]**2)
        print(kqiii)
        # calculate anharmonic cubic force constant
        # this will have nmodes rows and snmodes cols
        kqijj = np.divide(delfq_plus - 2.0 * delfq_zero + delfq_minus,
                          np.multiply(sel_delta, sel_delta).reshape(snmodes,1))
        print(kqijj.T[4])
        # get property values
        prop_grouped = prop.groupby('file')
        # get the property value for the equilibrium coordinate
        prop_zero = prop_grouped.get_group(0)
        prop_zero.drop(columns=['file'],inplace=True)
        prop_zero = np.repeat(prop_zero.values, snmodes)
        # get the property values for the positive displaced structures
        prop_plus = prop_grouped.filter(lambda x: x['file'].drop_duplicates().values in range(1,nmodes+1))
        prop_plus.drop(columns=['file'], inplace=True)
        prop_plus = prop_plus.values.reshape(snmodes,)
        # get the property values for the negative displaced structures
        prop_minus= prop_grouped.filter(lambda x: x['file'].drop_duplicates().values in
                                                                              range(nmodes+1, 2*nmodes+1))
        prop_minus.drop(columns=['file'], inplace=True)
        prop_minus = prop_minus.values.reshape(snmodes,)
        # generate the derivatives of the property
        dprop_dq = np.divide(prop_plus - prop_minus, 2*sel_delta)
        d2prop_dq2 = np.divide(prop_plus - 2*prop_zero + prop_minus, np.multiply(sel_delta, sel_delta))
        # done with setting up everything
        # moving on to the actual calculations

        #atom_frames = uni.atom['frame'].values
        #eqcoord = uni.atom.groupby('frame').get_group(atom_frames[-1])[['x','y','z']].values
        atom_order = eqcoord['symbol']
        coor_dfs = []
        zpvc_dfs = []
        va_dfs = []

        # calculate the ZPVC's at different temperatures by iterating over them
        for t in temperature:
            # calculate anharmonicity in the potential energy surface
            anharm = np.zeros(snmodes)
            for i in range(snmodes):
                temp1 = 0.0
                for j in range(nmodes):
                    # calculate the contribution of each vibration
                    temp_fac = self._get_temp_factor(t, frequencies[j])
                    # TODO: check the snmodes and nmodes indexing for kqijj
                    #       pretty sure that the rows are nmodes and the columns are snmodes
                    # TODO: check which is in the sqrt
                    # sum over the first index
                    temp1 += kqijj[j][i]/(frequencies[j]*rmass[j]*np.sqrt(sel_rmass[i]))*temp_fac
                # sum over the second index and set anharmonicity at each vibrational mode
                anharm[i] = -0.25*dprop_dq[i]/(sel_freq[i]**2*np.sqrt(sel_rmass[i]))*temp1
            # calculate curvature of property
            curva = np.zeros(snmodes)
            for i in range(snmodes):
                # calculate the contribution of each vibration
                temp_fac = self._get_temp_factor(t, sel_freq[i])
                # set the curvature at each vibrational mode
                curva[i] = 0.25*d2prop_dq2[i]/(sel_freq[i]*sel_rmass[i])*temp_fac

            # generate one of the zpvc dataframes
            va_dfs.append(pd.DataFrame.from_dict({'freq': sel_freq*Energy['Ha','cm^-1'], 'freqdx': select_freq,
                                                    'anharm': anharm, 'curva': curva, 'sum': anharm+curva,
                                                    'temp': np.repeat(t, snmodes)}))
            zpvc = np.sum(anharm+curva)
            tot_anharm = np.sum(anharm)
            tot_curva = np.sum(curva)
            zpvc_dfs.append([prop_zero[0], zpvc, prop_zero[0] + zpvc, tot_anharm, tot_curva, t])
            if print_results:
                print("========Results from Vibrational Averaging at {} K==========".format(t))
                # print results to stdout
                print("----Result of ZPVC calculation for {} of {} frequencies".format(snmodes, nmodes))
                print("    - Total Anharmonicity:   {:+.6f}".format(tot_anharm))
                print("    - Total Curvature:       {:+.6f}".format(tot_curva))
                print("    - Zero Point Vib. Corr.: {:+.6f}".format(zpvc))
                print("    - Zero Point Vib. Avg.:  {:+.6f}".format(prop_zero[0] + zpvc))
            if geometry:
                # calculate the effective geometry
                # we do not check this at the beginning as it will not always be computed
                #if not hasattr(uni, 'atom'):
                #    raise AttributeError("Please set the atom dataframe")
                sum_to_eff_geo = np.zeros((eqcoord.shape[0], 3))
                for i in range(snmodes):
                    temp1 = 0.0
                    for j in range(nmodes):
                        # calculate the contribution of each vibration
                        temp_fac = self._get_temp_factor(t, frequencies[j])
                        temp1 += kqijj[j][i]/(frequencies[j]*rmass[j]*np.sqrt(sel_rmass[i])) * temp_fac
                    # get the temperature correction to the geometry in Bohr
                    sum_to_eff_geo += -0.25 * temp1 / (sel_freq[i]**2 * np.sqrt(sel_rmass[i])) * \
                                        smat.groupby('freqdx').get_group(i)[['dx','dy','dz']].values
                # get the effective geometry
                tmp_coord = np.transpose(eqcoord[['x', 'y', 'z']].values + sum_to_eff_geo)
                # generate one of the coordinate dataframes
                # we write the frame to be the same as the temp column so that one can take
                # advantage of the exatomic.core.atom.Atom.to_xyz method
                coor_dfs.append(pd.DataFrame.from_dict({'set': list(range(len(eqcoord))),
                                                        'Z': atom_order.map(sym2z), 'x': tmp_coord[0],
                                                        'y': tmp_coord[1], 'z': tmp_coord[2],
                                                        'symbol': atom_order,
                                                        'temp': np.repeat(t, eqcoord.shape[0]),
                                                        'frame': np.repeat(t, len(eqcoord))}))
                # print out the effective geometry in Angstroms
                if print_results:
                    print("----Effective geometry in Angstroms")
                    xyz = coor_dfs[-1][['symbol','x','y','z']].copy()
                    xyz['x'] *= Length['au', 'Angstrom']
                    xyz['y'] *= Length['au', 'Angstrom']
                    xyz['z'] *= Length['au', 'Angstrom']
                    stargs = {'columns': None, 'header': False, 'index': False,
                              'formatters': {'symbol': '{:<5}'.format}, 'float_format': '{:6f}'.format}
                    print(xyz.to_string(**stargs))
        if geometry:
            self.eff_coord = pd.concat(coor_dfs, ignore_index=True)
        self.zpvc_results = pd.DataFrame(zpvc_dfs, columns=['property', 'zpvc', 'zpva', 'tot_anharm', 
                                                            'tot_curva', 'temp'])
        self.vib_average = pd.concat(va_dfs, ignore_index=True)

    def __init__(self, config_file, *args, **kwargs):
        config = Config.open_config(config_file, self._required_inputs,
                                    defaults=self._default_inputs)
        self.config = config

