import pandas as pd
import numpy as np
import os
from exatomic.core import Atom
from exa import TypedMeta
from exa.util.utility import mkp
from exa.util.units import Length

def gen_delta(freq, delta_type, disp=None, norm=0.04):
    """
    Function to compute the delta parameter to be used for the maximum distortion
    of the molecule along the normal mode.

    When delta_type = 1 we normalize the displacments to have a maximum of 0.04 Bohr
    on each normal mode.

    When delta_type = 0 we normalize all atomic displacements along all normal modes
    to have a global average displacement of 0.04 Bohr.

    When delta_type = 2 we normalize each displacement so the maximum displacement
    of any atom in the normal mode is 0.04 Bohr.

    When delta_typ = 3 the user can select a delta parameter to use with the disp
    keyword this will displace all normal modes by that delta parameter.

    Args:
        freq (:class:`exatomic.atom.Frequency`): Frequency dataframe
        delta_type (int): Integer value to define the type of delta parameter to use
    """
    nat = freq['label'].drop_duplicates().shape[0]
    freqdx = freq['freqdx'].unique()
    nmode = freqdx.shape[0]
    # global avrage displacement of 0.04 bohr for all atom displacements
    if delta_type == 0:
        d = np.sum(np.linalg.norm(
            freq[['dx', 'dy', 'dz']].values, axis=1))
        delta = norm * nat * nmode / (np.sqrt(3) * d)
        delta = np.repeat(delta, nmode)
    # average displacement of 0.04 bohr for each normal mode
    elif delta_type == 1:
        d = freq.groupby(['freqdx', 'frame']).apply(
            lambda x: np.sum(np.linalg.norm(
                x[['dx', 'dy', 'dz']].values, axis=1))).values
        delta = norm * nat / d
    # maximum displacement of 0.04 bohr for any atom in each normal mode
    elif delta_type == 2:
        d = freq.groupby(['freqdx', 'frame']).apply(lambda x:
            np.amax(abs(np.linalg.norm(x[['dx', 'dy', 'dz']].values, axis=1)))).values
        delta = norm / d
    elif delta_type == 3:
        if disp is not None:
            delta = np.repeat(disp, nmode)
        else:
            raise ValueError("Must provide a displacement value through the disp variable for " \
                             +"delta_type = 3")
    delta = pd.DataFrame.from_dict({'delta': delta, 'freqdx': freqdx})
    return delta

class DispMeta(TypedMeta):
    disp = Atom
    delta = pd.DataFrame
    atom = Atom

class Displace(metaclass=DispMeta):
    """
    Supporting class for Vibrational Averaging that will generate input files
    for a selected program under a certain displacement parameter.

    Computes displaced coordinates for all available normal modes from the equilibrium
    position by using the displacement vector components contained in the
    :class:`~exatomic.atom.Frequency` dataframe. It will scale these displacements to a
    desired type defined by the user with the delta_type keyword. For more information
    on this keyword see the documentation on the
    :class:`~exatomic.va.gen_delta` function.

    We can also define a specific normal mode or a list of normal modes that are of
    interest and generate displaced coordinates along those specific modes rather
    than all of the normal modes. This is highly recommended if applicable
    as it may reduce number of computations and memory usage significantly.

    Args:
        uni (:class:`~exatomic.Universe`): Universe object containg pertinent data
        delta_type (int): Integer value to define the type of delta parameter to use
        fdx (int or list): Integer or list parameter to only displace along the
                           selected normal modes
        disp (float): Floating point value to set a specific displacement delta
                      parameter. Must be used with delta_type=3
    """

    _tol = 1e-6

    def _gen_displaced(self, freq, atom_df, fdx):
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
        disp = freq_g[['dx','dy','dz']].values
        modes = freq_g.loc[unique_index, 'frequency'].values
        nat = eqcoord.shape[0]
        freqdx = freq_g['freqdx'].unique()
        tnmodes = freq['freqdx'].unique().shape[0]
        nmodes = freqdx.shape[0]
        # chop all values less than tolerance
        eqcoord[abs(eqcoord) < self._tol] = 0.0
        # get delta values for wanted frequencies
        try:
            if -1 in fdx:
                delta = self.delta['delta'].values
            elif -1 not in fdx:
                delta = self.delta.groupby('freqdx').filter(lambda x:
                                      fdx in x['freqdx'].drop_duplicates().values+1)['delta'].values
            else:
                raise TypeError("fdx must be a list of integers or a single integer")
            #if len(delta) != tnmodes:
            #    raise ValueError("Inappropriate length of delta. Passed a length of {} "+
            #                     "when it should have a length of {}. One value for each "+
            #                     "normal mode.".format(len(delta), tnmodes))
            #else:
            #    delta = np.repeat(delta, nat)
            delta = np.repeat(delta, nat)
        except AttributeError:
            raise AttributeError("Please compute self.delta first")
        # calculate displaced coordinates in positive and negative directions
        disp_pos = np.tile(np.transpose(eqcoord), nmodes) + np.multiply(np.transpose(disp), delta)
        disp_neg = np.tile(np.transpose(eqcoord), nmodes) - np.multiply(np.transpose(disp), delta)
        full = np.concatenate((eqcoord, np.transpose(disp_pos), np.transpose(disp_neg)), axis=0)
        # generate frequency index labels
        freqdx = [i+1+tnmodes*j for j in range(0,2,1) for i in freqdx]
        freqdx = np.concatenate(([0],freqdx))
        freqdx = np.repeat(freqdx, nat)
        # generate the modes column
        # useful if the frequency indexing is confusing
        modes = np.repeat(np.concatenate(([0],modes,modes)), nat)
        symbols = np.tile(symbols, 2*nmodes+1)
        znums = np.tile(znums, 2*nmodes+1)
        #frame = np.zeros(len(znums)).astype(np.int64)
        # create dataframe
        # coordinates are in units of Bohr as we use the coordinates from the atom dataframe
        df = pd.DataFrame(full, columns=['x', 'y', 'z'])
        df['freqdx'] = freqdx
        df['Z'] = znums
        df['symbol'] = symbols
        df['frequency'] = modes
        df['frame'] = freqdx
        return df

    @staticmethod
    def _write_data_file(path, array, fn):
        with open(mkp(path, fn), 'w') as f:
            for item in array:
                f.write("{}\n".format(item))

    def _create_data_files(self, uni, path=None, config=None):
        if path is None: path = os.getcwd()
        freq = uni.frequency.copy()
        atom = uni.atom.last_frame.copy()
        nat = atom.shape[0]
        fdxs = uni.frequency['freqdx'].drop_duplicates().index.values
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
            freq_ext = uni.frequency_ext.copy()
            redmass = freq_ext['r_mass'].values
        except AttributeError:
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
        # construct actual displacement data file
        fn = "displac_a.dat"
        rdelta = np.repeat(delta, nat)
        disp = np.multiply(np.linalg.norm(np.transpose(freq[['dx','dy','dz']].values), axis=0),
                                                        rdelta)
        disp *= Length['au', 'Angstrom']
        freqdx = freq['freqdx'].drop_duplicates().values
        n = len(atom_order)
        with open(mkp(path, fn), 'w') as f:
            f.write("actual displacement in angstroms\n")
            f.write("atom normal_mode distance_atom_moves\n")
            for fdx in range(len(freqdx)):
                for idx in range(n):
                    f.write("{} {}\t{}\n".format(idx+1, fdx+1, disp[fdx*nat+idx]))
        # construct initial configuration file
        template = "{:<20s}          {}\n".format
        text = ''
        text += template("DELTA_FILE", "delta.dat")
        text += template("SMATRIX_FILE", "smatrix.dat")
        text += template("ATOM_ORDER_FILE", "atom_order.dat")
        text += template("REDUCED_MASS_FILE", "redmass.dat")
        text += template("FREQUENCY_FILE", "freq.dat")
        text += template("DISPLAC_A_FILE", "displac_a.dat")
        if config is None:
            with open(mkp(path, 'va.conf'), 'w') as fn:
                fn.write(text)
        else:
            with open(mkp(path, config), 'a') as fn:
                fn.write(text)

    def __init__(self, uni, *args, **kwargs):
        if not hasattr(uni, 'frequency'):
            raise AttributeError("Frequency dataframe cannot be found in universe")
        delta_type = kwargs.pop("delta_type", 0)
        fdx = kwargs.pop("fdx", -1)
        disp = kwargs.pop("disp", None)
        norm = kwargs.pop("norm", 0.04)
        config = kwargs.pop("config", None)
        if isinstance(fdx, int):
            fdx = [fdx]
        freq = uni.frequency.copy()
        atom = uni.atom.copy()
        self.delta = gen_delta(freq, delta_type, disp, norm)
        self.disp = self._gen_displaced(freq, atom, fdx)
        self._create_data_files(uni, config=config)

