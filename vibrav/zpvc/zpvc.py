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
from vibrav.core import Config
from vibrav.util.print import dataframe_to_txt
from exatomic.exa.util.units import Length, Mass, Energy
from exatomic.util import conversions as conv
from exatomic.util.constants import Boltzmann_constant as boltzmann
from exatomic.core.atom import Atom
from exatomic.base import sym2z
import numpy as np
import pandas as pd
import os
import warnings

_zpvc_results = '''\
========Results from Vibrational Averaging at {temp:.2f} K==========
----Result of ZPVC calculation for {snmodes:d} of {nmodes:d} frequencies
    - Total Anharmonicity:   {anharm:+.6f}
    - Total Curvature:       {curva:+.6f}
    - Zero Point Vib. Corr.: {zpvc:+.6f}
    - Zero Point Vib. Avg.:  {zpva:+.6f}
'''

eff_geo = '''\
----Effective geometry in Angstroms for T={:.2f} K
{}'''

class ZPVC:
    '''
    Class to calculate the Zero-point vibrational corrections of a certain property.

    Required inputs in configuration file.

    +------------------+-----------------------------------------+--------------+
    | Attribute        | Description                             | Data Type    |
    +==================+=========================================+==============+
    | number_of_modes  | Number of normal modes in the molecule. | :obj:`int`   |
    +------------------+-----------------------------------------+--------------+
    | number_of_nuclei | Number of nuclei in the molecule.       | :obj:`int`   |
    +------------------+-----------------------------------------+--------------+
    | property_file    | Path to the CSV formatted file with the | :obj:`str`   |
    |                  | property information. Must have a       |              |
    |                  | header with the columns `'file'`,       |              |
    |                  | `'atom'` and the selected property      |              |
    |                  | column. In addition, it must have an    |              |
    |                  | index column.                           |              |
    +------------------+-----------------------------------------+--------------+
    | gradient_file    | Path to the CSV formatted file with the | :obj:`str`   |
    |                  | gradient information. Must have a       |              |
    |                  | header with the columns                 |              |
    |                  | `['file', 'atom', 'fx', 'fy', 'fz']`.   |              |
    |                  | In addition, it must have an index      |              |
    |                  | column.                                 |              |
    +------------------+-----------------------------------------+--------------+
    | property_atoms   | Atomic index of the atom/s of interest. | :obj:`int`   |
    |                  | Can be given as multiple space          |              |
    |                  | separated integers.                     |              |
    +------------------+-----------------------------------------+--------------+
    | property_column  | Column name with the data of interest   | :obj:`str`   |
    |                  | in the property file.                   |              |
    +------------------+-----------------------------------------+--------------+
    | temperature      | Temperature/s in Kelvin. Can be given   | :obj:`float` |
    |                  | as a space separated list of floats.    |              |
    +------------------+-----------------------------------------+--------------+

    Default inputs in the configuration file.

    +-----------------+------------------------------------------+----------------+
    | Attribute       | Description                              | Default Value  |
    +=================+==========================================+================+
    | smatrix_file    | Filepath containing all of the           | smatrix.dat    |
    |                 | information of the normal mode           |                |
    |                 | displacements.                           |                |
    +-----------------+------------------------------------------+----------------+
    | eqcoord_file    | Filepath containing the coordinates of   | eqcoord.dat    |
    |                 | the equilibrium structure.               |                |
    +-----------------+------------------------------------------+----------------+
    | atom_order_file | Filepath containing the atomic symbols   | atom_order.dat |
    |                 | and ordering of the nuclei.              |                |
    +-----------------+------------------------------------------+----------------+

    We implement the equations as outlined in the paper
    *J. Phys. Chem. A* 2005, **109**, 8617-8623
    (doi:`10.1021/jp051685y <https://doi.org/10.1021/jp051685y>`_).
    Where we can compute the Zero-Point Vibrational corrections with

    .. math::
        \\text{ZPVC} = -\\frac{1}{4}\\sum_{i=1}^m\\frac{1}{\\omega_i^2\\sqrt{\\mu_i}}
                        \\left(\\frac{\\partial P}{\\partial Q_i}\\right)
                        \\sum_{j=1}^m\\frac{k_{ijj}}{\\omega_j\\mu_j\\sqrt{\\mu_i}}
                       +\\frac{1}{4}\sum_{i=1}^m\\frac{1}{\\omega_i\\mu_i}
                        \\left(\\frac{\\partial^2 P}{\\partial Q_i^2}\\right)

    Where, :math:`m` represents the total number of normal modes,
    :math:`\\omega_i` is the frequency of the :math:`i`th normal mode,
    and :math:`\\mu_i` is the reduced mass, in atomic units. The
    derivatives of the property (:math:`P`) are taken with respect to
    the normal coordinates, :math:`Q_i`, for a given normal mode
    :math:`i`. The anharmonic cubic constant, :math:`k_{ijj}`, is
    defined as the mixed third-energy derivative, and calculated as,

    .. math::
        k_{ijj} = \\frac{\\partial^3 E}{\\partial Q_i \\partial Q_j^2}

    The calculated energy gradients in terms of the normal modes can be
    obtained from the Cartesian gradients by,

    .. math::
        \\frac{\\partial E_{+/0/-}}{\\partial Q_i} =
            \\sum_{\\alpha=1}^{3n} \\frac{\\partial E_{+/0/-}}{\\partial x_{\\alpha}}
                S_{\\alpha j}

    '''
    _required_inputs = {'number_of_modes': int, 'number_of_nuclei': int,
                        'property_file': str, 'gradient_file': str,
                        'property_atoms': (list, int), 'property_column': str,
                        'temperature': (list, float)}
    _default_inputs = {'smatrix_file': ('smatrix.dat', str),
                       'eqcoord_file': ('eqcoord.dat', str),
                       'atom_order_file': ('atom_order.dat', str),
                       'index_col': (True, bool)}

    @staticmethod
    def _get_temp_factor(temp, freq):
        if temp > 1e-6:
            try:
                factor = freq*conv.Ha2J / (2 * boltzmann * temp)
                temp_fac = np.cosh(factor) / np.sinh(factor)
            # this should be taken care of by the conditional but always good to
            # take care of explicitly
            except ZeroDivisionError:
                raise ZeroDivisionError("Something seems to have gone wrong " \
                                        +"with the sinh function")
        else:
            temp_fac = 1.
        return temp_fac

    @staticmethod
    def _check_file_continuity(df, prop, nmodes):
        '''
        Make sure that the input data frame has a positive and negative
        displacement.

        Note:
            The input data frame must have a column named `'file'`.

        Args:
            df (:class:`pandas.DataFrame`): Data frame containing the
                data of interest. Must have a `'file'` column.
            prop (:obj:`str`): String for better debugging to know
                which data frame it failed on.
            nmodes (:obj:`int`): Number of normal modes.

        Returns:
            rdf (:class:`pandas.DataFrame`): Data frame after extracting
                the frequencies that are missing data points.
        '''
        files = df['file'].drop_duplicates()
        pos_file = files[files.isin(range(1,nmodes+1))]
        neg_file = files[files.isin(range(nmodes+1, 2*nmodes+1))]-nmodes
        intersect = np.intersect1d(pos_file.values, neg_file.values)
        diff = np.unique(np.concatenate((np.setdiff1d(pos_file.values, intersect),
                                         np.setdiff1d(neg_file.values, intersect)), axis=None))
        rdf = df.copy()
        if len(diff) > 0:
            msg = "Seems that we are missing one of the {} outputs for frequency {} " \
                  +"we will ignore the {} data for these frequencies."
            print(msg.format(prop, diff, prop))
            rdf = rdf[~rdf['file'].isin(diff)]
            rdf = rdf[~rdf['file'].isin(diff+nmodes)]
        return rdf

    @staticmethod
    def get_pos_neg_gradients(grad, freq, nmodes):
        '''
        Calculate the energy gradients in terms of the respective normal
        modes for the given displacement.

        Note:
            The input data frames `grad` and `freq` must have the columns
            `['fx', 'fy', 'fz']` and `['dx', 'dy', 'dz']`, respectively.

        Args:
            grad (:class:`pandas.DataFrame`): DataFrame containing all
                of the gradient data
            freq (:class:`pandas.DataFrame`): DataFrame containing all
                of the frequency data

        Returns:
            delfq_zero (:class:`pandas.DataFrame`): Normal mode
                converted gradients of equilibrium structure
            delfq_plus (:class:`pandas.DataFrame`): Normal mode
                converted gradients of positive displaced structure
            delfq_minus (:class:`pandas.DataFrame`): Normal mode
                converted gradients of negative displaced structure
        '''
        grouped = grad.groupby('file')
        # generate delta dataframe
        # TODO: make something so delta can be set
        #       possible issues are a user using a different type of delta
        #nmodes = len(smat)
        # get gradient of the equilibrium coordinates
        grad_0 = grouped.get_group(0)
        # get gradients of the displaced coordinates in the positive direction
        grad_plus = grouped.filter(lambda x: x['file'].unique() in range(1,nmodes+1))
        # get gradients of the displaced coordinates in the negative direction
        grad_minus = grouped.filter(lambda x: x['file'].unique() in range(nmodes+1, 2*nmodes+1))
        # TODO: Check if we can make use of numba to speed up this code
        delfq_zero = freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda x:
                                    np.sum(np.multiply(grad_0[['fx', 'fy', 'fz']].values,
                                                       x.values))).values
        # we extend the size of this 1d array as we will perform some matrix summations with the
        # other outputs from this method
        delfq_zero = np.tile(delfq_zero, nmodes).reshape(nmodes, nmodes)
        delfq_plus = grad_plus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values)))).values
        delfq_minus = grad_minus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                                freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                    np.sum(np.multiply(y.values, x.values)))).values
        return [delfq_zero, delfq_plus, delfq_minus]

    @staticmethod
    def calculate_frequencies(delfq_plus, delfq_minus, redmass, nmodes, delta):
        '''
        Here we calculated the frequencies from the gradients calculated
        for each of the displaced structures along the normal mode. In
        principle this should give the same or nearly the same frequency
        value as that from a frequency calculation.

        Args:
            delfq_0 (numpy.ndarray): Array that holds all of the
                information about the gradient derivative of the
                equlilibrium coordinates.
            delfq_plus (numpy.ndarray): Array that holds all of the
                information about the gradient derivative of the
                positive displaced coordinates.
            delfq_minus (numpy.ndarray): Array that holds all of the
                information about the gradient derivative of the
                negative displaced coordinates.
            redmass (numpy.ndarray): Array that holds all of the
                reduced masses. We can handle both a subset of the
                entire values or all of the values.
            select_freq (numpy.ndarray): Array that holds the selected
                frequency indexes.
            delta (numpy.ndarray): Array that has the delta values used
                in the displaced structures.

        Returns:
            frequencies (numpy.ndarray): Frequency array from the
                calculation.
        '''
        # calculate force constants
        kqi = np.zeros(nmodes)
        #print(redmass_sel.shape)
        for fdx in range(nmodes):
            kqi[fdx] = (delfq_plus[fdx][fdx] - delfq_minus[fdx][fdx]) / (2.0*delta[fdx])
        vqi = np.divide(kqi, redmass.reshape(nmodes,))
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
        frequencies = np.sqrt(vqi).reshape(nmodes,)*conv.Ha2inv_cm
        return frequencies

    def zpvc(self, geometry=True, print_results=False,
             write_out_files=True, debug=False):
        """
        Method to compute the Zero-Point Vibrational Corrections.

        Args:
            geometry (:obj:`bool`, optional): Bool value that tells the
                program to also calculate the effective geometry.
                Defaults to :code:`True`.
            print_results (:obj:`bool`, optional): Bool value to print
                the results from the zpvc calcualtion to stdout.
                Defaults to :code:`False`.
            write_out_files (:obj:`bool`, optional): Bool value to
                write files with the final results to a CSV formatted
                and txt file. Defaults to :code:`True`.
            debug (:obj:`bool`, optional): Bool value to write extra
                matrices with debug information to a file including the
                gradients expressed in terms of the normal modes, and
                the first and second derivatives of the property.

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
        if write_out_files:
            if not os.path.exists(zpvc_dir):
                os.mkdir(zpvc_dir)
        config = self.config
        if config.index_col:
            gradient = pd.read_csv(config.gradient_file, index_col=0) \
                                    .sort_values(by=['file', 'atom'])
            property = pd.read_csv(config.property_file, index_col=0) \
                                    .sort_values(by=['file', 'atom'])
        else:
            gradient = pd.read_csv(config.gradient_file, index_col=False) \
                                    .sort_values(by=['file', 'atom'])
            property = pd.read_csv(config.property_file, index_col=False) \
                                    .sort_values(by=['file', 'atom'])
        grouped = property.groupby('atom').get_group
        temperature = config.temperature
        pcol = config.property_column
        coor_dfs = []
        zpvc_dfs = []
        va_dfs = []
        for atom in config.property_atoms:
            prop_vals = grouped(atom)[[pcol, 'file']]
            if prop_vals.shape[1] != 2:
                raise ValueError("Property dataframe must have a second dimension of 2 not " \
                                 +"{}".format(property.shape[1]))
            # get the total number of normal modes
            nmodes = config.number_of_modes
            if prop_vals.shape[0] != 2*nmodes+1:
                raise ValueError("The number of entries in the property data frame must " \
                                 +"be twice the number of normal modes plus one, currently " \
                                 +"{}".format(property.shape[0]))
            # check for any missing files and remove the respective counterpart
            grad = self._check_file_continuity(gradient.copy(), 'gradient', nmodes)
            prop = self._check_file_continuity(prop_vals.copy(), 'property', nmodes)
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
            if snmodes < nmodes:
                raise NotImplementedError("We do not currently have support to handle missing frequencies")
            # get the actual frequencies
            # TODO: check if we should use the real or calculated frequencies
            frequencies = pd.read_csv(config.frequency_file, header=None).values.reshape(-1,)
            frequencies *= conv.inv_cm2Ha
            if any(frequencies < 0):
                text = "Negative frequencies were found in {}. Make sure that the geometry " \
                       +"optimization and frequency calculations proceeded correctly."
                warnings.warn(text.format(config.frequency_file, Warning))
            rmass = pd.read_csv(config.reduced_mass_file, header=None).values.reshape(-1,)
            rmass /= conv.amu2u
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
            # check the gradients by calculating the freuqncies numerically
            num_freqs = self.calculate_frequencies(delfq_plus, delfq_minus, rmass, nmodes, delta)
            # calculate anharmonic cubic force constant
            # this will have nmodes rows and nmodes cols
            kqijj = []
            for i in range(nmodes):
                kqijj.append((delfq_plus[i] - 2.0*delfq_zero[i] + delfq_minus[i]) / delta[i]**2)
            kqijj = np.array(kqijj)
            # get the cubic force constant
            kqiii = np.diagonal(kqijj)
            # get property values
            prop_grouped = prop.groupby('file')
            # get equil property
            prop_zero = prop_grouped.get_group(0)[config.property_column].values
            prop_zero = np.repeat(prop_zero, nmodes)
            # positive displacement
            prop_plus = prop_grouped.filter(lambda x: x['file'].unique() in range(1, nmodes+1))
            prop_plus = prop_plus.sort_values(by=['file'])[config.property_column].values.flatten()
            # negative displacement
            prop_minus = prop_grouped.filter(lambda x: x['file'].unique() in range(nmodes+1, 2*nmodes+1))
            prop_minus = prop_minus.sort_values(by=['file'])[config.property_column].values.flatten()
            # calculate derivatives
            dprop_dq = (prop_plus - prop_minus) / (2*delta)
            d2prop_dq2 = (prop_plus - 2*prop_zero + prop_minus) / (delta**2)
            # write output files for comparisons
            if write_out_files:
                # write the anharmonic cubic constant matrix
                fp = os.path.join(zpvc_dir, 'kqijj')
                df = pd.DataFrame(kqijj)
                df.columns.name = 'cols'
                df.index.name = 'rows'
                df.to_csv(fp+'.csv')
                dataframe_to_txt(df=df, ncols=4, fp=fp+'.txt')
                # write the cubic force constant
                fp = os.path.join(zpvc_dir, 'kqiii')
                df = pd.DataFrame(kqiii.reshape(1,-1))
                df.to_csv(fp+'.csv')
                dataframe_to_txt(df=df, ncols=4, fp=fp+'.txt')
                if debug:
                    # write the first derivative of the property
                    fp = os.path.join(zpvc_dir, 'dprop-dq')
                    df = pd.DataFrame([dprop_dq]).T
                    df.index.name = 'freqdx'
                    df.to_csv(fp+'.csv')
                    dataframe_to_txt(df=df, fp=fp+'.txt')
                    # write the second derivative of the property
                    fp = os.path.join(zpvc_dir, 'd2prop-dq2')
                    df = pd.DataFrame([d2prop_dq2]).T
                    df.index.name = 'freqdx'
                    df.to_csv(fp+'.csv')
                    dataframe_to_txt(df=df, fp=fp+'.txt')
                    # write the gradients in terms of the normal modes
                    # equilibrium structure
                    fp = os.path.join(zpvc_dir, 'delfq-zero')
                    df = pd.DataFrame(delfq_zero)
                    df.to_csv(fp+'.csv')
                    dataframe_to_txt(df=df, fp=fp+'.txt')
                    # positive displacments
                    fp = os.path.join(zpvc_dir, 'delfq-plus')
                    df = pd.DataFrame(delfq_plus)
                    df.to_csv(fp+'.csv')
                    dataframe_to_txt(df=df, fp=fp+'.txt')
                    # negative displacements
                    fp = os.path.join(zpvc_dir, 'delfq-minus')
                    df = pd.DataFrame(delfq_minus)
                    df.to_csv(fp+'.csv')
                    dataframe_to_txt(df=df, fp=fp+'.txt')
            # done with setting up everything
            # moving on to the actual calculations
            atom_order = eqcoord['symbol']
            # calculate the ZPVC's at different temperatures by iterating over them
            for tdx, t in enumerate(temperature):
                # calculate anharmonicity in the potential energy surface
                anharm = np.zeros(snmodes)
                for i in range(nmodes):
                    temp1 = 0.0
                    for j in range(nmodes):
                        # calculate the contribution of each vibration
                        temp_fac = self._get_temp_factor(t, frequencies[j])
                        # sum over the first index
                        temp1 += kqijj[j][i]/(frequencies[j]*rmass[j] \
                                              *np.sqrt(rmass[i]))*temp_fac
                    # sum over the second index and set anharmonicity at
                    # each vibrational mode
                    anharm[i] = -0.25*dprop_dq[i]/(frequencies[i]**2 \
                                                    *np.sqrt(rmass[i]))*temp1
                # calculate curvature of property
                curva = np.zeros(snmodes)
                for i in range(nmodes):
                    # calculate the contribution of each vibration
                    temp_fac = self._get_temp_factor(t, frequencies[i])
                    # set the curvature at each vibrational mode
                    curva[i] = 0.25*d2prop_dq2[i]/(frequencies[i]*rmass[i])*temp_fac
                # generate one of the zpvc dataframes
                dict_df = dict(frequency=frequencies*conv.Ha2inv_cm,
                               num_frequency=num_freqs, freqdx=range(nmodes),
                               anharm=anharm, curva=curva, sum=anharm+curva,
                               temp=np.repeat(t, nmodes),
                               atom=np.repeat(atom, nmodes),
                               frame=np.repeat(tdx, nmodes))
                va_dfs.append(pd.DataFrame.from_dict(dict_df))
                zpvc = np.sum(anharm+curva)
                tot_anharm = np.sum(anharm)
                tot_curva = np.sum(curva)
                zpvc_dfs.append([prop_zero[0], zpvc, prop_zero[0] + zpvc, tot_anharm,
                                 tot_curva, t, atom, tdx])
                if print_results:
                    print(_zpvc_results.format(temp=t, snmodes=snmodes, nmodes=nmodes,
                                               anharm=tot_anharm, curva=tot_curva,
                                               zpvc=zpvc, zpva=prop_zero[0]+zpvc))
        for tdx, t in enumerate(temperature):
            if geometry:
                # calculate the effective geometry
                # we do not check this at the beginning as it will not always be computed
                sum_to_eff_geo = np.zeros((eqcoord.shape[0], 3))
                for i in range(snmodes):
                    temp1 = 0.0
                    for j in range(nmodes):
                        # calculate the contribution of each vibration
                        temp_fac = self._get_temp_factor(t, frequencies[j])
                        temp1 += kqijj[j][i]/(frequencies[j]*rmass[j]*np.sqrt(rmass[i])) * temp_fac
                    # get the temperature correction to the geometry in Bohr
                    sum_to_eff_geo += -0.25 * temp1 / (frequencies[i]**2 * np.sqrt(rmass[i])) * \
                                        smat.groupby('freqdx').get_group(i)[['dx','dy','dz']].values
                # get the effective geometry
                tmp_coord = np.transpose(eqcoord[['x', 'y', 'z']].values + sum_to_eff_geo)
                # generate one of the coordinate dataframes
                # we write the frame to be the same as the temp column so that one can take
                # advantage of the exatomic.core.atom.Atom.to_xyz method
                df = pd.DataFrame.from_dict({'set': list(range(len(eqcoord))),
                                             'Z': atom_order.map(sym2z), 'x': tmp_coord[0],
                                             'y': tmp_coord[1], 'z': tmp_coord[2],
                                             'symbol': atom_order,
                                             'temp': np.repeat(t, eqcoord.shape[0]),
                                             'frame': np.repeat(tdx, len(eqcoord))})
                cols = ['x', 'y', 'z']
                for col in cols:
                    df.loc[df[col].abs() < 1e-6, col] = 0
                df = Atom(df)
                coor_dfs.append(df)
                # print out the effective geometry in Angstroms
                if print_results:
                    print(eff_geo.format(t, df.to_xyz()))
        if geometry:
            self.eff_coord = Atom(pd.concat(coor_dfs, ignore_index=True))
            if write_out_files:
                fp_temp = 'atomic-coords-{:03d}.xyz'
                for frame in range(self.eff_coord.nframes):
                    fp = os.path.join(zpvc_dir, fp_temp.format(frame))
                    comment = 'Vibrational averaged positions for T={:.2f} K'
                    with open(fp, 'w') as fn:
                        t = self.eff_coord.groupby('frame') \
                                .get_group(frame)['temp'].unique()[0]
                        kwargs = dict(header=True, frame=frame,
                                      comments=comment.format(t))
                        text = self.eff_coord.to_xyz(**kwargs)
                        fn.write(text)
        cols = ['property', 'zpvc', 'zpva', 'tot_anharm',
                'tot_curva', 'temp', 'atom', 'frame']
        # save data as class attributes
        self.zpvc_results = pd.DataFrame(zpvc_dfs, columns=cols)
        self.vib_average = pd.concat(va_dfs, ignore_index=True)
        if write_out_files:
            # write the zpvc results to file
            formatters = ['{:12.5f}'.format] + ['{:12.7f}'.format]*4 \
                         + ['{:9.3f}'.format] + ['{:8d}'.format, '{:4d}'.format]
            fp = os.path.join(zpvc_dir, 'results')
            self.zpvc_results.to_csv(fp+'.csv')
            dataframe_to_txt(self.zpvc_results, ncols=6, fp=fp+'.txt',
                             float_format=formatters)
            # write the full vibrational average table to file
            formatters = ['{:10.3f}'.format]*2+['{:8d}'.format] \
                         + ['{:12.7f}'.format]*3 + ['{:9.3f}'.format] \
                         + ['{:8d}'.format, '{:4d}'.format]
            fp = os.path.join(zpvc_dir, 'vibrational-average')
            self.vib_average.to_csv(fp+'.csv')
            dataframe_to_txt(self.vib_average, ncols=6, fp=fp+'.txt',
                             float_format=formatters)

    def __init__(self, config_file, *args, **kwargs):
        config = Config.open_config(config_file, self._required_inputs,
                                    defaults=self._default_inputs)
        config.temperature = tuple(sorted(config.temperature))
        self.config = config

