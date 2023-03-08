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
import warnings
import os
from exatomic.core.atom import Atom
from exatomic.exa.core.container import TypedMeta
from exatomic.exa.util.units import Length

def gen_delta(freq, delta_type, disp=None, norm=None):
    """
    Function to compute the delta parameter to be used for the maximum distortion
    of the molecule along the normal mode.

    When :code:`delta_type = 0` we normalize all atomic displacements along all normal modes
    to have a global average displacement given by the :code:`norm` parameter.

    When :code:`delta_type = 1` we normalize the displacments to have a maximum given by the
    :code:`norm` parameter on each normal mode.

    When :code:`delta_type = 2` we normalize each displacement so the maximum displacement
    vector of any atom in the normal mode is set to the :code:`norm` parameter.

    When :code:`delta_type = 3` the user can select a constant delta parameter to use with the
    disp keyword this will displace all normal modes by that delta parameter.

    Args:
        freq (:class:`exatomic.atom.Frequency`): Frequency dataframe.
        delta_type (:obj:`int`): Integer value to define the type of delta parameter to use.
        disp (:obj:`float`, optional): Constant displacement parameter to use with
                                       `delta_type=3`. Defaults to :code:`None`
        norm (:obj:`float`, optional): Normalization parameter to use. Defaults to 0.04 au.

    Returns:
        delta (:obj:`pandas.DataFrame`): Delta parameters to multiply into each normal mode.

    Raises:
        ValueError: When :code:`delta_type=3` and :code:`disp=None`.

    Examples:
    """
    if not type(norm) == list:
        norm = [norm]
    data = freq.copy()
    nat = data['label'].drop_duplicates().shape[0]
    freqdx = data['freqdx'].unique()
    nmode = freqdx.shape[0]
    deltas = []
    for n in norm:
        # global avrage displacement of 0.04 bohr for all atom displacements
        if delta_type == 0:
            d = np.sum(np.linalg.norm(
                data[['dx', 'dy', 'dz']].values, axis=1))
            delta = n * nat * nmode / (np.sqrt(3) * d)
            delta = np.repeat(delta, nmode)
        # average displacement of 0.04 bohr for each normal mode
        elif delta_type == 1:
            d = data.groupby(['freqdx', 'frame']).apply(
                lambda x: np.sum(np.linalg.norm(
                    x[['dx', 'dy', 'dz']].values, axis=1))).values
            delta = n * nat / d
        # maximum displacement of 0.04 bohr for any atom in each normal mode
        elif delta_type == 2:
            d = data.groupby(['freqdx', 'frame']).apply(lambda x:
                np.amax(abs(np.linalg.norm(x[['dx', 'dy', 'dz']].values, axis=1)))).values
            delta = n / d
        elif delta_type == 3:
            if disp is not None:
                delta = np.repeat(disp, nmode)
            else:
                raise ValueError("Must provide a displacement value through the disp variable for " \
                                 +"delta_type = 3")
        delta = pd.DataFrame.from_dict({'delta': delta, 'freqdx': freqdx})
        delta['norm'] = n
        deltas.append(delta)
    delta = pd.concat(deltas, ignore_index=True)
    return delta

class DispMeta(TypedMeta):
    disp = Atom
    delta = pd.DataFrame
    atom = Atom

class Displace(metaclass=DispMeta):
    """
    Supporting class for Vibrational Averaging that will displace the input atomic
    coordinates along its normal modes (:func:`~vibrav.Displace.gen_displaced`).

    Computes displaced coordinates for all available normal modes from the equilibrium
    position by using the displacement vector components contained in the
    :class:`~exatomic.atom.Frequency` dataframe. It will scale these displacements to a
    desired type defined by the user with the delta_type keyword. For more information
    on this keyword see the documentation on the

    We can also define a specific normal mode or a list of normal modes that are of
    interest and generate displaced coordinates along those specific modes rather
    than all of the normal modes. This is highly recommended if applicable
    as it may reduce number of computations and memory usage significantly.

    Args:
        cls (:class:`~exatomic.Universe`): Universe object containg pertinent data
        delta_type (int): Integer value to define the type of delta parameter to use
        fdx (int or list): Integer or list parameter to only displace along the
                           selected normal modes
        disp (float): Floating point value to set a specific displacement delta
                      parameter. Must be used with delta_type=3
        mwc (bool): Divide the normal modes by the reduced mass prior to calculating
                    the delta parameter
    """

    _tol = 1e-6

    @staticmethod
    def _insert_vals(df, znums, symbols, fdx, frame, freq,
                     label, norm, delta):
        df['Z'] = znums
        df['symbol'] = symbols
        df['freqdx'] = fdx
        df['frame'] = frame
        df['frequency'] = freq
        df['label'] = label
        df['norm'] = norm
        df['delta'] = delta

    def gen_displaced(self, freq, atom_df, fdx):
        """
        Function to generate displaced coordinates for each selected normal mode.
        We scale the displacements by the selected delta value in the positive and negative
        directions. We generate an array of coordinates that are put into a dataframe to
        write them to a file input for later evaluation.

        Note:
            The index 0 is reserved for the optimized coordinates, the equilibrium geometry.
            The displaced coordinates in the positive direction are given an index from
            1 to tnmodes (total number of normal modes).
            The displaced coordinates in the negative direction are given an index from
            tnmodes to 2*tnmodes.
            In an example with 39 normal modes 0 is the equilibrium geometry, 1-39 are the
            positive displacements and 40-78 are the negative displacements.
            nmodes are the number of normal modes that are selected. tnmodes are the total
            number of normal modes for the system.

        Args:
            freq (:class:`exatomic.atom.Frequency`): Frequency dataframe
            atom (:class:`exatomic.atom.Atom`): Atom dataframe
            fdx (int or list): Integer or list parameter to only displace along the
                               selected normal modes
        """
        # get needed data from dataframes
        # atom coordinates should be in Bohr
        atom = atom_df.last_frame
        eqcoord = atom[['x', 'y', 'z']].values
        symbols = atom['symbol'].values
        # gaussian Fchk class uses Zeff where the Output class uses Z
        # add try block to account for the possible exception
        try:
            znums = atom['Zeff'].values
        except KeyError:
            znums = atom['Z'].values
        if -1 in fdx:
            freq_g = freq.copy()
        else:
            freq_g = freq.groupby('freqdx').filter(lambda x: fdx in
                                                    x['freqdx'].drop_duplicates().values+1).copy()
        unique_index = freq_g['freqdx'].drop_duplicates().index
        modes = freq_g.loc[unique_index, 'frequency'].values
        nat = eqcoord.shape[0]
        freqdx = freq_g['freqdx'].unique()
        tnmodes = freq['freqdx'].unique().shape[0]
        nmodes = freqdx.shape[0]
        grouped = freq_g.groupby('freqdx')
        cols = ['dx', 'dy', 'dz']
        displaced = atom[['x', 'y', 'z']].copy()
        self._insert_vals(displaced, znums, symbols, 0, 0,
                          0.0, range(nat), 0.0, 0.0)
        dfs = []
        arr = enumerate(self.delta.groupby(['norm', 'freqdx']))
        for idx, ((norm, fdx), delta_df) in arr:
            delta = delta_df['delta'].values[0]
            disp = grouped.get_group(fdx)[cols].values
            frame = int(idx/nmodes)*2*nmodes
            # create the positive displacement coordinates
            disp_pos = eqcoord + disp*delta
            df = pd.DataFrame(disp_pos, columns=['x', 'y', 'z'])
            self._insert_vals(df, znums, symbols, fdx+1, fdx+1+frame,
                              modes[fdx], range(nat), norm, delta)
            dfs.append(df)
            # create the negative displacement coordinates
            disp_neg = eqcoord - disp*delta
            df = pd.DataFrame(disp_neg, columns=['x', 'y', 'z'])
            self._insert_vals(df, znums, symbols, fdx+nmodes+1,
                              fdx+nmodes+1+frame, modes[fdx],
                              range(nat), norm, delta)
            dfs.append(df)
        displaced = pd.concat([displaced]+dfs, ignore_index=True)
        displaced.sort_values(by=['frame', 'label'], inplace=True)
        displaced.reset_index(drop=True, inplace=True)
        nnorms = self.delta.norm.unique().shape[0]
        print("Making sure that the displacements and order is correct")
        for fdx in range(2*nmodes):
            sign = 1 if fdx < nmodes else -1
            idxs = [fdx+1+x*2*nmodes for x in range(nnorms)]
            df = displaced.groupby('frame').filter(lambda x: x['frame'].unique() in idxs)
            cols = ['x', 'y', 'z']
            coords = np.tile(atom[cols].values.flatten(),
                             nnorms).reshape(nnorms*nat, 3)
            df = (df[cols].values - coords)/df[['delta']].values
            df *= sign
            df[abs(df) < 1e-6] = 0.0
            df = pd.DataFrame(df)
            df['group'] = np.repeat(range(nnorms), nat)
            f = fdx if fdx < nmodes else fdx - nmodes
            disps = freq_g.groupby('freqdx').get_group(f)[['dx', 'dy', 'dz']].values
            for group, data in df.groupby('group'):
                close = np.allclose(data[range(3)].values, disps, atol=1e-6)
                if not close:
                    print("Difference between displaced and real modes")
                    print(data[range(3)].values - disps)
                    print("Elements that are close to 1e-6")
                    print(np.isclose(data[range(3)].values, disps, atol=1e-6))
                    msg = "There was an issue with the displacement {}"
                    raise ValueError(msg.format(group*nmodes+fdx+1))
        print("Done with check")
        return Atom(displaced)

    def gen_displaced_cartesian(self, atom_df, delta=0.01, include_zeroth=True,
                                exclude=None):
        """
        Function to generate displaced coordinates in cartesian space. Will generate
        a total of 3N displacements with N being the number of atoms. If you desire
        the functionality to displace along the calculated normal modes please see
        :func:`vibrav.util.gen_displaced.Displace.gen_displaced`. We use the
        convention where nat*3 + j + nat*3*sign
        """
        # get needed data from dataframes
        # atom coordinates should be in Bohr
        atom = atom_df.last_frame
        eqcoord = atom[['x', 'y', 'z']].values
        symbols = atom['symbol'].values
        nat = atom.shape[0]
        delta_au = delta*Length['Angstrom', 'au']
        # gaussian Fchk class uses Zeff where the Output class uses Z
        # add try block to account for the possible exception
        try:
            znums = atom['Zeff'].values
        except KeyError:
            try:
                znums = atom['Z'].values
            except KeyError:
                pass
        # chop all values less than tolerance
        eqcoord[abs(eqcoord) < 1e-6] = 0.0
        modes = np.eye(3)
        disp = []
        for idx in range(nat):
            for jdx, vec in enumerate(modes):
                for ndx, sign in enumerate([1, -1]):
                    coord = eqcoord.copy()
                    if exclude is not None:
                        if (idx, jdx) != exclude:
                            coord[idx] += sign*vec*delta_au
                    else:
                        coord[idx] += sign*vec*delta_au
                    df = pd.DataFrame(coord, columns=['x', 'y', 'z'])
                    df['symbol'] = symbols
                    df['frame'] = idx*3+jdx+ndx*3*nat+1
                    df['set'] = range(nat)
                    disp.append(df)
        if include_zeroth:
            df = pd.DataFrame(eqcoord, columns=['x', 'y', 'z'])
            df['symbol'] = symbols
            df['frame'] = 0
            df['set'] = range(nat)
            disp.append(df)
        disp = Atom(pd.concat(disp, ignore_index=True))
        disp.sort_values(by=['frame', 'set'], inplace=True)
        disp.reset_index(drop=True, inplace=True)
        return disp

    @staticmethod
    def _write_data_file(path, array, fn):
        with open(os.path.join(path, fn), 'w') as f:
            for item in array:
                f.write("{}\n".format(item))

    def create_data_files(self, freq, atom, norm, cart_disp, disp,
                          path=None, config=None):
        '''
        Method to create the .dat files that are needed to perform the calculations for
        vibrational averaging.

        This script will auto-generate the configuration file (:code:`'va.conf'` by default)
        given in the config parameter. It will create it with the following input options,

        - **DELTA_FILE**: delta.dat
        - **SMATRIX_FILE**: smatrix.dat
        - **ATOM_ORDER_FILE**: atom_order.dat
        - **REDUCED_MASS_FILE**: redmass.dat
        - **FREQUENCY_FILE**: freq.dat
        - **EQCOORD_FILE**: eqcoord.dat

        Args:
            cls (:class:`exatomic.Universe`): Universe object that has the frequency and atom
                                              dataframes.
            path (:obj:`str`, optional): Path to save the files to. Defaults to `None`.
            config (:obj:`str`, optional): Path to base config file. Defaults to `None`.
        '''
        if path is None: path = os.getcwd()
        nat = atom.shape[0]
        fdxs = freq['freqdx'].drop_duplicates().index.values
        nmodes = fdxs.shape[0]
        nnorms = len(norm)
        # construct delta data file
        fn = "delta.dat"
        delta = self.delta['delta'].values
        self._write_data_file(path=path, array=delta, fn=fn)
        # construct smatrix data file
        fn = "smatrix.dat"
        smatrix = freq[['dx', 'dy', 'dz']].stack().values
        self._write_data_file(path=path, array=smatrix, fn=fn)
        # construct atom order data file
        fn = "atom_order.dat"
        atom_order = atom['symbol'].values
        self._write_data_file(path=path, array=atom_order, fn=fn)
        # construct reduced mass data file
        fn = "redmass.dat"
        try:
            if freq['r_mass'].shape[0] > 0:
                redmass = freq.loc[fdxs, 'r_mass'].values
        except KeyError:
            raise AttributeError("Could not find the reduced masses in either the frequency " \
                                 +"dataframe and could not find the frequency_ext dataframe.")
        self._write_data_file(path=path, array=redmass, fn=fn)
        # construct eqcoord data file
        fn = "eqcoord.dat"
        eqcoord = atom[['x', 'y', 'z']].stack().values
        eqcoord *= Length['au', 'Angstrom']
        self._write_data_file(path=path, array=eqcoord, fn=fn)
        # construct frequency data file
        fn = "freq.dat"
        frequency = freq.loc[fdxs, 'frequency'].values
        self._write_data_file(path=path, array=frequency, fn=fn)
        ## construct actual displacement data file
        #fn = "displac_a.dat"
        #rdelta = np.repeat(delta, nat)
        #disp = np.multiply(np.linalg.norm(np.transpose(freq[['dx','dy','dz']].values), axis=0),
        #                                                rdelta)
        #disp *= Length['au', 'Angstrom']
        #freqdx = freq['freqdx'].drop_duplicates().values
        #n = len(atom_order)
        #with open(os.path.join(path, fn), 'w') as f:
        #    f.write("actual displacement in angstroms\n")
        #    f.write("atom normal_mode distance_atom_moves\n")
        #    for fdx in range(len(freqdx)):
        #        for idx in range(n):
        #            f.write("{} {}\t{}\n".format(idx+1, fdx+1, disp[fdx*nat+idx]))
        # construct initial configuration file
        template = "{:<20s}          {}\n".format
        text = ''
        text += template("DELTA_FILE", "delta.dat")
        text += template("SMATRIX_FILE", "smatrix.dat")
        text += template("ATOM_ORDER_FILE", "atom_order.dat")
        text += template("REDUCED_MASS_FILE", "redmass.dat")
        text += template("FREQUENCY_FILE", "freq.dat")
        #text += template("DISPLAC_A_FILE", "displac_a.dat")
        text += template("EQCOORD_FILE", "eqcoord.dat")
        text += template("NUMBER_OF_NUCLEI", nat)
        text += template("NUMBER_OF_MODES", nmodes)
        text += template("NUMBER_OF_NORMS", nnorms)
        if not cart_disp:
            text += template("NORM_FACTORS",
                             ' '.join(list(map(str, norm))))
        else:
            text += template("NORM_FACTORS",
                             ' '.join(list(map(str, [disp]))))
        text += template("CARTESIAN_DISP", int(cart_disp))
        if config is None:
            with open(os.path.join(path, 'va.conf'), 'w') as fn:
                fn.write(text)
        else:
            with open(os.path.join(path, config), 'a') as fn:
                fn.write(text)

    def __init__(self, cls, *args, **kwargs):
        delta_type = kwargs.pop("delta_type", 0)
        fdx = kwargs.pop("fdx", -1)
        disp = kwargs.pop("disp", None)
        norm = kwargs.pop("norm", [0.04])
        if not isinstance(norm, (list, tuple)):
            norm = [norm]
        norm = sorted(norm)
        config = kwargs.pop("config", None)
        mwc = kwargs.pop("mwc", False)
        path = kwargs.pop("path", None)
        cart_disp = kwargs.pop("cart_disp", False)
        atom_file = kwargs.pop("atom_file", None)
        freq_file = kwargs.pop("freq_file", None)
        write_files = kwargs.pop("write_files", True)
        csv_props = kwargs.pop("csv_props", False)
        if not csv_props:
            atom = cls.atom.copy()
            freq = cls.frequency.copy()
        else:
            from exatomic.core import Atom, Frequency
            atom = Atom(pd.read_csv(atom_file))
            freq = Frequency(pd.read_csv(freq_file))
        if isinstance(fdx, int):
            fdx = [fdx]
        if not cart_disp:
            if mwc:
                freq[['dx', 'dy', 'dz']] /= np.sqrt(freq['r_mass'].values).reshape(-1,1)
                text = "We are dividing the normal modes by the sqrt of the " \
                       +"reduced mass. This is not implemented into any " \
                       +"of the scripts in VIBRAV and is untested."
                warnings.warn(text, Warning)
            self.delta = gen_delta(freq, delta_type, disp, norm)
            self.disp = self.gen_displaced(freq, atom, fdx)
        else:
            nat = atom.last_frame.shape[0]
            delta_dict = dict(delta=[disp*Length['Angstrom', 'au']]*nat)
            self.delta = pd.DataFrame.from_dict(delta_dict)
            self.disp = self.gen_displaced_cartesian(atom, disp)
        if write_files:
            self.create_data_files(atom=atom.last_frame, freq=freq, config=config,
                                   norm=norm, path=path, cart_disp=cart_disp,
                                   disp=disp)

