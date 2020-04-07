from exa.core import Editor, TypedMeta
from exa.util.units import Length, Mass, Energy
from exatomic.core import Atom, Frequency
from exatomic.base import z2sym, sym2isomass
import numpy as np
import pandas as pd
import six
import warnings

class Tape21Meta(TypedMeta):
    atom = Atom
    frequency = Frequency

class Tape21(six.with_metaclass(Tape21Meta, Editor)):
    '''
    Parser for ADF Tape21 that have been converted to an ASCII file with
    their dmpkf utility.
    '''
    def _intme(self, fitem, idx=0):
        return int(self[fitem[idx]].split()[-1])

    def _dfme(self, fitem, dim, idx=0):
        start = fitem[idx] + 2
        col = min(len(self[start].split()), dim)
        stop = np.ceil(start + dim / col).astype(np.int64)
        return self.pandas_dataframe(start, stop, col).stack().values

    def parse_frequency(self):
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
        _recartmodes = "Normalmodes"
        _refreq = "Frequencies"
        found = self.find(_refreq, _renorm, _recartmodes, keys_only=True)
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
        print(freq)
        nmodes = freq.shape[0]
        freq = np.repeat(freq, nat)
        if found[_renorm]:
            # get the mass-weighted normal modes
            ndisps = int(self[found[_renorm][0]+1].split()[0])
            normalmodes = self._dfme(np.array(found[_renorm]), ndisps, idx=0)
            calc_rmass = True
        elif found[_recartnorm] and not found[_renorm]:
            # get the non-mass-weighted normal modes and toss warning
            text = "Mass-weighted normal modes could not be found. If you are " \
                  +"performing vibrational analysis they must be mass-weighted " \
                  +"normal modes."
            warnings.warn(text, Warning)
            ndisps = int(self[found[_recartmodes][0]+1].split()[0])
            normalmodes = self._dfme(np.array(found[_recartmodes]), ndisps, idx=0)
            calc_rmass = False
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
        # put the data together
        df = pd.DataFrame({'dx': dx, 'dy': dy, 'dz': dz, 'frequency': freq,
                           'freqdx': freqdx})
        cols = ['dx', 'dy', 'dz']
        # calculate the reduced masses
        if calc_rmass:
            r_mass = df.groupby(['freqdx']).apply(lambda x: 1 \
                                                        / np.sum(np.square(x[cols].values.flatten()) \
                                                        * 1/mass)).values
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

