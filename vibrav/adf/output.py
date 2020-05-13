from exa.core import Editor, TypedMeta
from exa.util.units import Length, Mass, Energy
from exatomic.core.atom import Atom, Frequency
from exatomic.core.gradient import Gradient
from exatomic.core.tensor import JCoupling
from exatomic.base import z2sym, sym2isomass
import numpy as np
import pandas as pd
import six
import warnings

class Tape21Meta(TypedMeta):
    atom = Atom
    frequency = Frequency
    gradient = Gradient
    j_coupling = JCoupling

class Tape21(six.with_metaclass(Tape21Meta, Editor)):
    '''
    Parser for ADF Tape21 that have been converted to an ASCII file with
    their dmpkf utility.
    '''

    @staticmethod
    def rmass_mwc(data, symbol):
        cols = ['dx', 'dy', 'dz']
        mapper = sym2isomass(symbol)
        mass = list(map(mapper.get, symbol))
        mass = np.repeat(mass, 3).astype(float)
        mass = mass.reshape(data[cols].shape)
        disps = data[cols].values
        r_mass = np.sum(np.square(disps)/mass)
        r_mass = 1/r_mass
        return r_mass

    @staticmethod
    def rmass_cart(data, symbol):
        cols = ['dx', 'dy', 'dz']
        mapper = sym2isomass(symbol)
        mass = list(map(mapper.get, symbol))
        mass = np.repeat(mass, 3).astype(float)
        mass = mass.reshape(data[cols].shape)
        disps = data[cols].values
        norms = np.linalg.norm(disps*np.sqrt(mass))
        norms = 1/norms
        disps *= norms
        r_mass = np.sum(np.square(disps))
        r_mass = 1/r_mass
        return r_mass

    def _intme(self, fitem, idx=0):
        return int(self[fitem[idx]+1].split()[0])

    def _dfme(self, fitem, dim, idx=0):
        start = fitem[idx] + 2
        col = min(len(self[start].split()), dim)
        stop = np.ceil(start + dim / col).astype(np.int64)
        return self.pandas_dataframe(start, stop, col).stack().values

    def parse_frequency(self, cart=True):
        '''
        ADF frequency parser.

        Note:
            This will toss a warning if it cannot find the mass-weighted normal modes
            which must be used to generate the displaced structures for vibrational
            averaging. Also, it will be unable to calculate the reduced masses as it will
            have normalized cartesian coordinates where it expects normalized
            mass-weighted cartesian normal modes.
        '''
        # search flags
        _renorm = "NormalModes_RAW"
        _recartnorm = "Normalmodes"
        _refreq = "Frequencies"
        _refreqexc = r"\bFrequencies\b"
        _rekey = r"\bFreq\b"
        found = self.find(_refreq, _renorm, _recartnorm, keys_only=True)
        key = self.regex(_rekey, _refreqexc, keys_only=True)
        # need to do this to ensure that we only match the data in the Freq block
        found_freq = []
        for k in key[_rekey]:
            for f in found[_refreq]:
                if f-1 == k:
                    found_freq.append(f)
                    break
        if not found_freq:
            return
        found[_refreq] = found_freq
        if not found[_refreq]:
            return
        if not hasattr(self, 'atom'):
            self.parse_atom()
        # get the number of atoms
        nat = self.atom.last_frame.shape[0]
        # get the frequencies
        freq = self._dfme(found[_refreq], nat*3)
        # find where the frequencies are zero
        # these should be the ones that ADF determines to be translations and rotations
        # TODO: need a test case with one imaginary frequency
        low = freq != 0
        nlow = low.shape[0]
        # get only the ones that are non-zero
        freq = freq[low]
        nmodes = freq.shape[0]
        freq = np.repeat(freq, nat)
        if found[_renorm] and not cart:
            # get the mass-weighted normal modes
            ndisps = int(self[found[_renorm][0]+1].split()[0])
            normalmodes = self._dfme(np.array(found[_renorm]), ndisps, idx=0)
            calc_rmass = True
        elif found[_recartnorm] and cart:
            # get the non-mass-weighted normal modes and toss warning
            ndisps = int(self[found[_recartnorm][0]+1].split()[0])
            normalmodes = self._dfme(np.array(found[_recartnorm]), ndisps, idx=0)
            calc_rmass = True
        else:
            raise Exception("Something went wrong")
        # get the vibrational modes in the three cartesian directions
        # the loop is neede in case there are any negative modes
        # because then the normal mode displacements for the negative mode
        # are listed first and we need those
        dx = []
        dy = []
        dz = []
        for idx in np.where(low)[0]:
            dx.append(normalmodes[idx*nat*3+0:(idx+1)*nat*3+0:3])
            dy.append(normalmodes[idx*nat*3+1:(idx+1)*nat*3+1:3])
            dz.append(normalmodes[idx*nat*3+2:(idx+1)*nat*3+2:3])
        # flatten arrays to vectors
        dx = np.array(dx).flatten()
        dy = np.array(dy).flatten()
        dz = np.array(dz).flatten()
        freqdx = np.repeat(range(nmodes), nat)
        label = np.tile(self.atom['label'], nmodes)
        symbol = self.atom['symbol']
        # get the isotopic masses
        mapper = sym2isomass(symbol)
        mass = symbol.map(mapper).astype(float).values
        symbol = np.tile(self.atom['symbol'], nmodes)
        mass = np.repeat(mass, 3)
        print(dx.shape, dy.shape, dz.shape, freq.shape, freqdx.shape)
        # put the data together
        df = pd.DataFrame({'dx': dx, 'dy': dy, 'dz': dz, 'frequency': freq,
                           'freqdx': freqdx})
        cols = ['dx', 'dy', 'dz']
        # calculate the reduced masses
        if not cart:
            r_mass = df.groupby(['freqdx']).apply(self.rmass_mwc, self.atom['symbol']).values
        else:
            r_mass = df.groupby(['freqdx']).apply(self.rmass_cart, self.atom['symbol']).values
        df['r_mass'] = np.repeat(r_mass, nat)
        df['symbol'] = symbol
        df['label'] = label
        df['ir_int'] = 0
        df['frame'] = 0
        self.frequency = df

    def parse_atom(self):
        # search flags
        _reatom = "xyz InputOrder"
        _reqtch = "qtch"
        _rentyp = "ntyp"
        _renqptr = "nqptr"
        _reinporder = "atom order index"
        _remass = "mass"
        found = self.find(_reatom, _reqtch, _rentyp, _renqptr, _reinporder, _remass, keys_only=True)
        ncoords = int(self[found[_reatom][0]+1].split()[0])
        coords = self._dfme(np.array(found[_reatom]), ncoords)
        x = coords[::3]
        y = coords[1::3]
        z = coords[2::3]
        # get the number of atom types
        ntyp = int(self[found[_rentyp][1]+2].split()[0])
        # get the charges for each atom type
        qtch = self._dfme(found[_reqtch], ntyp)
        # get the span of each atom type
        nqptr = self._dfme(found[_renqptr], ntyp+1) - 1
        nat = nqptr.max()
        # get the znum vector from the ordered atom table
        zordered = np.zeros(nat)
        for n in range(ntyp):
            for idx in range(nqptr[n], nqptr[n+1]):
                zordered[idx] = qtch[n]
        # convert to the input structure
        zinput = np.zeros(nat)
        input_order = self._dfme(found[_reinporder], nat*2).reshape(2, nat).astype(int) - 1
        # iterate over the input order array as this gives the location of each atom type after
        # the re-ordering done in adf
        for od, inp in zip(input_order[0], range(nat)):
            zinput[inp] = zordered[od]
        set = np.array(list(range(nat)))
        symbol = pd.Series(zinput).map(z2sym)
        # put it all together
        df = pd.DataFrame.from_dict({'symbol': symbol, 'set': set, 'label': set, 'x': x, 'y': y,
                                     'z': z, 'Z': zinput, 'frame': 0})
        self.atom = df

    def parse_gradient(self):
        # search flags
        _reinpgrad = "Gradients_InputOrder"
        _refrggrad = "Gradients_CART"
        found = self.find(_reinpgrad, _refrggrad, keys_only=True)
        if not found[_reinpgrad]:
            raise NotImplementedError("Have not implemented reading the re-ordered gradients.")
        # get the gradients
        ngrad = self._intme(np.array(found[_reinpgrad]))
        grad = self._dfme(np.array(found[_reinpgrad]), ngrad)
        x = grad[::3]
        y = grad[1::3]
        z = grad[2::3]
        if not hasattr(self, 'atom'):
            self.parse_atom()
        symbol = self.atom['symbol'].values
        Z = self.atom['Z'].values.astype(int)
        atom = list(range(len(x)))
        df = pd.DataFrame.from_dict({'Z': Z, 'atom': atom, 'fx': x, 'fy': y, 'fz': z, 'symbol': symbol,
                                     'frame': 0})
        df = df[['atom', 'Z', 'fx', 'fy', 'fz', 'symbol', 'frame']]
        #for u in ['fx', 'fy', 'fz']: df[u] *= 1./Length['Angstrom', 'au']
        self.gradient = df

    def parse_nmr_shielding(self):
        raise NotImplementedError("Coming soon!!")

    def parse_j_coupling(self):
        _reiso = "NMR Coupling J const InputOrder"
        _retensor = "NMR Coupling J tens InputOrder"
        found = self.find(_reiso, _retensor, keys_only=True)
        if not found[_reiso]:
            return
        if not hasattr(self, 'atom'):
            self.parse_atom()
        ncoupl = self._intme(found[_reiso])
        natom = np.sqrt(ncoupl)
        coupling = self._dfme(found[_reiso], ncoupl)
        ntens = self._intme(found[_retensor])
        tensor = self._dfme(found[_retensor], ntens)
        requested = np.where(coupling != 0)[0]
        tensor = tensor.reshape(ncoupl, 9)[requested]
        cols = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        df = pd.DataFrame(tensor, columns=cols)
        atoms = np.transpose(list(map(lambda x: divmod(x, natom), requested)))
        df['isotropic'] = coupling[coupling != 0]
        df['atom'] = atoms[0].astype(int)
        symbols = self.atom['symbol'].values
        if len(symbols) > natom:
            raise NotImplementedError("Cannot deal with more than one atom frame.")
        df['symbol'] = list(map(lambda x: symbols[x], df['atom'].values))
        df['pt_atom'] = atoms[1].astype(int)
        df['pt_symbol'] = list(map(lambda x: symbols[x], df['pt_atom'].values))
        df['label'] = 'j_coupling'
        df['frame'] = 0
        self.j_coupling = df

