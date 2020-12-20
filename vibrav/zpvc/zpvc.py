from exa import logging
logging.disable()
from vibrav.core import Config
from vibrav.util.print import dataframe_to_txt
from exa.util.units import Length, Mass, Energy
from exa.util.constants import Boltzmann_constant as boltzmann
from exatomic.core.atom import Atom
from exatomic.base import sym2z
import numpy as np
import pandas as pd
import os
import warnings

class ZPVC:
    '''
    Class to calculate the Zero-point vibrational corrections of a certain property.

    Required inputs in configuration file.

    +------------------+-----------------------------------------+------------+
    | Attribute        | Description                             | Data Type  |
    +==================+=========================================+============+
    | number_of_modes  | Number of normal modes in the molecule. | :obj:`int` |
    +------------------+-----------------------------------------+------------+
    | number_of_nuclei | Number of nuclei in the molecule.       | :obj:`int` |
    +------------------+-----------------------------------------+------------+

    Default inputs in the configuration file.

    +-----------------+-----------------------------------------------------+----------------+
    | Attribute       | Description                                         | Default Value  |
    +=================+=====================================================+================+
    | smatrix_file    | Filepath containing all of the information of the   | smatrix.dat    |
    |                 | normal mode displacements.                          |                |
    +-----------------+-----------------------------------------------------+----------------+
    | eqcoord_file    | Filepath containing the coordinates of the          | eqcoord.dat    |
    |                 | equilibrium structure.                              |                |
    +-----------------+-----------------------------------------------------+----------------+
    | atom_order_file | Filepath containing the atomic symbols and ordering | atom_order.dat |
    |                 | of the nuclei.                                      |                |
    +-----------------+-----------------------------------------------------+----------------+
    '''
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
        outlined in the paper *J. Phys. Chem. A* 2005, **109**, 8617-8623 (doi:10.1021/jp051685y).
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
            gradient (:class:`pandas.DataFrame`): Data frame of the gradients from the calculations.
            property (:class:`pandas.DataFrame`): Data frame of the properties that were calculated.
            temperature (:obj:`list`, optional): List object containing all of the temperatures of
                                                 interest. Defaults to :code:`None` (sets temperature
                                                 to 0).
            geometry (:obj:`bool`, optional): Bool value that tells the program to also calculate the
                                              effective geometry. Defaults to :code:`True`.
            print_results(:obj:`bool`, optional): Bool value to print the results from the zpvc
                                                  calcualtion to stdout. Defaults to :code:`False`.

        Examples:
            This example will use some of the resource files that are used in the tests. This is
            supposed to be a minimalistic example to explain the different objects that are
            generated by this method. For a full example starting from generating the displaced
            structures to extraction and analysis of the data refer to **insert link to example**.

            We will start by importing all of the necessary packages.

            >>> from vibrav.base import resource
            >>> import pandas as pd
            >>> import numpy as np
            >>> import tarfile
            >>> import glob
            >>> import os

            Now extract all of the :code:`'*.dat'` files in the resource tarball.

            >>> with tarfile.open(resource('nitromalinamide-zpvc-dat-files.tar.xz'),
            ...                   'r:xz') as tar:
            ...     tar.extractall()
            ... 
            >>> 

            Now we have all of the data files that are needed by the configuration script to run
            the :meth:`vibrav.zpvc.ZPVC.zpvc` method. Now we define some constants for easier
            typing.

            >>> nat = 15
            >>> nmodes = 3*15 - 6
            >>> disp = nmodes*2 + 1

            Next we get the gradient and property data from the resource files. Note, the gradient
            resource file is a single column array. The code is expecting a data frame with four
            columns with the column names :code:`['fx', 'fy', 'fz', 'file']`. This can also be a
            :class:`exatomic.core.atom.Gradient` object. The property data must be a two column
            data frame where one of the columns has the column label :code:`'label'`. The other
            column is not required to have a specific name.

            >>> df = pd.read_csv(resource('nitromalonamide-zpvc-grad.dat.xz'), header=None
            ...                  compression='xz')
            >>> tmp = df.values.reshape(nat*disp, 3)
            >>> grad = pd.DataFrame(tmp, columns=['fx', 'fy', 'fz'])
            >>> grad['file'] = np.repeat(range(disp), nat)
            >>> prop = pd.read_csv(resource('nitromalonamide-zpvc-prop.dat.xz'), header=None,
            ...                    compression='xz')
            >>> prop['file'] = prop.index

            Now we have all of the pieces together to be able to run the ZPVC code. All that is
            left is to initiate the class instance.

            >>> zpvc = ZPVC(config_file=resource('nitromalonamide-zpvc-config.conf'))

            Now we run the ZPVC code.

            >>> zpvc.zpvc(gradient=grad, property=prop, temperature=[0])

            For now we only run this for one temperature to simplify the output. We now have the
            class attributes :attr:`zpvc_results` and :attr:`vib_average`. Let us print out the
            :attr:`vib_average` class attribute.

            >>> print(zpvc.vib_average.to_string())
                     freq  freqdx    anharm     curva       sum  temp
            0     85.0284       0  0.000000  0.007649  0.007649     0
            1     89.3751       1 -0.000000 -0.001816 -0.001816     0
            2    146.1814       2 -0.000000  0.022266  0.022266     0
            3    217.6824       3 -0.000000  0.001199  0.001199     0
            4    320.8656       4 -1.079300  0.076429 -1.002872     0
            5    354.5547       5 -0.171530  0.010393 -0.161137     0
            6    401.8652       6  0.000964  0.002991  0.003955     0
            7    418.5204       7  0.000000 -0.012716 -0.012716     0
            8    425.1060       8 -0.102758  0.009911 -0.092847     0
            9    433.7701       9  0.000000 -0.022071 -0.022071     0
            10   461.1199      10 -0.314054  0.074176 -0.239878     0
            11   485.3209      11  0.000522  0.003446  0.003968     0
            12   609.5199      12 -0.090530  0.009844 -0.080687     0
            13   666.6248      13  0.000000 -0.002926 -0.002926     0
            14   685.1025      14 -0.000000 -0.006440 -0.006440     0
            15   703.9068      15 -0.362860  0.028033 -0.334827     0
            16   714.8914      16  0.000000  0.002550  0.002550     0
            17   725.8531      17  0.000000 -0.005967 -0.005967     0
            18   762.7554      18 -0.000000 -0.005501 -0.005501     0
            19   846.2045      19 -0.003575 -0.000841 -0.004416     0
            20  1075.3188      20 -0.018814  0.004701 -0.014112     0
            21  1094.5938      21 -0.032282  0.005943 -0.026339     0
            22  1107.2629      22  0.000000  0.046784  0.046784     0
            23  1161.5631      23 -0.011198  0.002291 -0.008907     0
            24  1174.0458      24 -0.020984  0.009741 -0.011243     0
            25  1265.5979      25 -0.531411  0.183772 -0.347639     0
            26  1316.6681      26 -0.003551  0.002776 -0.000775     0
            27  1395.3206      27 -0.001487  0.003657  0.002170     0
            28  1452.1272      28 -0.002516 -0.001166 -0.003682     0
            29  1555.7496      29  0.000756 -0.000758 -0.000002     0
            30  1575.8570      30  0.003893 -0.001330  0.002563     0
            31  1598.3631      31 -0.003270 -0.012490 -0.015761     0
            32  1631.1443      32 -0.039127  0.011868 -0.027259     0
            33  1710.9460      33 -0.054030  0.009701 -0.044329     0
            34  2255.7412      34 -0.906408  0.403475 -0.502933     0
            35  3520.1806      35 -0.000387  0.000993  0.000606     0
            36  3541.9146      36 -0.002419 -0.000220 -0.002639     0
            37  3686.9440      37 -0.001406  0.000354 -0.001052     0
            38  3696.3968      38 -0.000614 -0.000124 -0.000738     0

            The information in the columns of the table shown above is as follows:

            - freq: Frequencies of the normal modes. By default these are the same as the input
              frequencies from the data in the :code:`'frequency_file'`.
            - freqdx: Frequency indeces in a zero-based python format indexing.
            - anharm: The anharmonicity of the potential energy surface given by,
            .. math::
                    \\Delta P_1 = -\\frac{1}{4} \\sum_{i=1}^{m} \\frac{1}{\\omega_i^2\\sqrt{\\mu_i}}
                                  \\left(\\frac{\\partial P}{\\partial Q_i}\\right) \\sum_{j=1}^{m}
                                  \\frac{k_{ijj}}{\\omega_j \\mu_j \\sqrt{\\mu_i}}


            - curva: The curvature of the property surface given by,
            .. math::
                    \\Delta P_2 = \\frac{1}{4} \\sum_{i=1}{m} \\frac{1}{\\omega_i \\mu_i}
                                    \\left(\\frac{\\partial^2 P}{\\partial Q_i^2}\\right)


            - sum: Summation of the anharmonicity and curvature values.
            - temp: Temperature at which the calculation was run.

            Now we print the contents of the :attr:`zpvc_results` attribute.

            >>> print(zpvc.zpvc_results.to_string())
                property      zpvc      zpva  tot_anharm  tot_curva  temp
            0  13.932882 -2.887802  11.04508   -3.748377   0.860575     0

            The information in the columns above is as follows:

            - property: The property calculated at the equilibrium coordinates.
            - zpvc: The Zero-Point vibrational correction to be applied on the property.
            - zpva: The Zero-Point averaged property values. Summation of the :code:`'property'`
            and :code:`'zpvc'` columns.
            - tot_anharm: Summation of all of the normal mode contributions to the anharmonicity.
            - tot_curva: Summation of all of the normal mode contributions to the curvature.
            - temp: Temperature at which the calculation was run.

            Now we can print the contents of the :attr:`eff_coord` attribute.

            >>> print(zpvc.eff_coord.to_string())
                set  Z         x         y             z symbol  temp  frame
            0     0  1  0.343242 -2.168186  3.205383e-09      H     0      0
            1     1  1 -3.122279 -1.159822 -3.496945e-07      H     0      0
            2     2  1 -2.603661  0.536159 -9.922190e-09      H     0      0
            3     3  1  3.307589 -0.342839  1.244102e-08      H     0      0
            4     4  1  2.390190  1.172662  2.278307e-08      H     0      0
            5     5  6  1.315911 -0.538146 -1.324874e-09      C     0      0
            6     6  8  1.403388 -1.824836 -1.411113e-08      O     0      0
            7     7  7  2.456096  0.168373  1.482704e-08      N     0      0
            8     8  6 -0.009395  0.061711 -1.115198e-09      C     0      0
            9     9  6 -1.142284 -0.865999  1.072109e-09      C     0      0
            10   10  8 -0.887831 -2.122312  8.894732e-09      O     0      0
            11   11  7 -0.187867  1.468788 -1.620257e-09      N     0      0
            12   12  8  0.810723  2.219691 -1.043966e-08      O     0      0
            13   13  8 -1.342229  1.946239  7.251946e-09      O     0      0
            14   14  7 -2.421288 -0.456714 -4.873884e-09      N     0      0

            All of the columns are defined in :class:`exatomic.core.atom.Atom`.

            The final thing is to see what happens when we run the calculation for more than
            one temperature.

            >>> zpvc.zpvc(gradient=grad, property=prop, temperature=[0, 100, 200])
            >>> print(zpvc.zpvc_results.to_string())
                property      zpvc       zpva  tot_anharm  tot_curva  temp
            0  13.932882 -2.887802  11.045080   -3.748377   0.860575     0
            1  13.932882 -2.804743  11.128138   -3.678383   0.873640   100
            2  13.932882 -2.648282  11.284599   -3.570384   0.922102   200

            The only thing that has changed from the example above is that we have added more
            rows to the :attr:`zpvc_results` attribute. This will be the case for all of the
            class attributes printed above.
        """
        zpvc_dir = 'zpvc-outputs'
        if not os.path.exists(zpvc_dir): os.mkdir(zpvc_dir)
        config = self.config
        if property.shape[1] != 2:
            raise ValueError("Property dataframe must have a second dimension of 2 not " \
                             +"{}".format(property.shape[1]))
        if temperature is None: temperature = [0]
        # get the total number of normal modes
        nmodes = config.number_of_modes
        if property.shape[0] != 2*nmodes+1:
            raise ValueError("The number of entries in the property data frame must " \
                             +"be twice the number of normal modes plus one, currently " \
                             +"{}".format(property.shape[0]))
        # check for any missing files and remove the respective counterpart
        grad = self._check_file_continuity(gradient.copy(), 'gradient', nmodes)
        prop = self._check_file_continuity(property.copy(), 'property', nmodes)
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
        if any(frequencies < 0):
            text = "Negative frequencies were found in {}. Make sure that the geometry " \
                   +"optimization and frequency calculations proceeded correctly."
            warnings.warn(text.format(config.frequency_file, Warning))
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
        fp = os.path.join(zpvc_dir, 'kqiii')
        df = pd.DataFrame(kqiii.reshape(1,-1))
        df.to_csv(fp+'.csv')
        dataframe_to_txt(df=df, ncols=4, fp=fp+'.txt')
        # calculate anharmonic cubic force constant
        # this will have nmodes rows and snmodes cols
        kqijj = np.divide(delfq_plus - 2.0 * delfq_zero + delfq_minus,
                          np.multiply(sel_delta, sel_delta).reshape(snmodes,1))
        fp = os.path.join(zpvc_dir, 'kqijj')
        df = pd.DataFrame(kqijj)
        df.columns.name = 'cols'
        df.index.name = 'rows'
        df.to_csv(fp+'.csv')
        dataframe_to_txt(df=df, ncols=4, fp=fp+'.txt')
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
        fp = os.path.join(zpvc_dir, 'dprop-dq')
        df = pd.DataFrame(dprop_dq.reshape(1, -1))
        df.columns.name = 'frequency'
        df.to_csv(fp+'.csv')
        dataframe_to_txt(df=df, ncols=4, fp=fp+'.txt')
        d2prop_dq2 = np.divide(prop_plus - 2*prop_zero + prop_minus, np.multiply(sel_delta, sel_delta))
        fp = os.path.join(zpvc_dir, 'd2prop-dq2')
        df = pd.DataFrame(dprop_dq.reshape(1, -1))
        df.columns.name = 'frequency'
        df.to_csv(fp+'.csv')
        dataframe_to_txt(df=df, ncols=4, fp=fp+'.txt')
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
        formatters = ['{:12.5f}'.format] + ['{:12.7f}'.format]*4 + ['{:d}'.format]
        fp = os.path.join(zpvc_dir, 'results')
        self.zpvc_results.to_csv(fp+'.csv')
        dataframe_to_txt(self.zpvc_results, ncols=6, fp=fp+'.txt', float_format=formatters)
        self.vib_average = pd.concat(va_dfs, ignore_index=True)
        formatters = ['{:10.3f}'.format, '{:8d}'.format] + ['{:12.7f}'.format]*3 + ['{:5d}'.format]
        fp = os.path.join(zpvc_dir, 'vibrational-average')
        self.vib_average.to_csv(fp+'.csv')
        dataframe_to_txt(self.vib_average, ncols=6, fp=fp+'.txt', float_format=formatters)

    def __init__(self, config_file, *args, **kwargs):
        config = Config.open_config(config_file, self._required_inputs,
                                    defaults=self._default_inputs)
        self.config = config

