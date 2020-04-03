from exa.core import Editor, TypedMeta
from exa.util.units import Length, Mass, Energy
from exatomic.core import Atom, Frequency
from exatomic.base import z2sym, sym2isomass
import numpy as np
import pandas as pd
import six

class Tape21Meta(TypedMeta):
    atom = Atom
    frequency = Frequency

class Tape21(six.with_metaclass(Tape21Meta, Editor)):
    def _intme(self, fitem, idx=0):
        return int(self[fitem[idx]].split()[-1])

    def _dfme(self, fitem, dim, idx=0):
        start = fitem[idx] + 2
        col = min(len(self[start].split()), dim)
        stop = np.ceil(start + dim / col).astype(np.int64)
        return self.pandas_dataframe(start, stop, col).stack().values

    def parse_frequency(self):
        # search flags
        _renorm = "NormalModes_RAW"
        _refreq = "Frequencies"
        found = self.find(_refreq, keys_only=True)
        if not found:
            return
        found = self.find(_refreq, _renorm, keys_only=True)
        if not hasattr(self, 'atom'):
            self.parse_atom()
        nat = self.atom.shape[0]
        freq = self._dfme(found[_refreq], nat*3)
        low = np.where(freq == 0)[0]
        nlow = low.shape[0]
        freq = freq[nlow:]
        nmodes = freq.shape[0]
        freq = np.repeat(freq, nat)
        ndisps = int(self[found[_renorm][0]+1].split()[0])
        normalmodes = self._dfme(np.array(found[_renorm]), ndisps, idx=0)
        dx = normalmodes[nlow*nat*3::3]
        dy = normalmodes[nlow*nat*3+1::3]
        dz = normalmodes[nlow*nat*3+2::3]
        freqdx = np.repeat(range(nmodes), nat)
        label = np.tile(self.atom['label'], nmodes)
        symbol = self.atom['symbol']
        mapper = sym2isomass(symbol)
        mass = symbol.map(mapper).astype(float).values
        symbol = np.tile(self.atom['symbol'], nmodes)
        #mass = sym2isomass(self.atom['symbol'].values)
        mass = np.repeat(mass, 3)
        df = pd.DataFrame({'dx': dx, 'dy': dy, 'dz': dz, 'frequency': freq,
                           'freqdx': freqdx})
        cols = ['dx', 'dy', 'dz']
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

