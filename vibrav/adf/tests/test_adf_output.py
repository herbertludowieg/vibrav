from vibrav import adf
from vibrav.base import resource
import lzma
import os
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def editor(res_file):
    comp = resource(res_file)
    decomp = comp.split(os.sep)[-1][:-3]
    with open(decomp, 'wb') as new_file, lzma.LZMAFile(comp, 'rb') as file:
        for data in iter(lambda : file.read(100 *1024), b''):
            new_file.write(data)
    editor = adf.Tape21(decomp)
    yield editor
    os.remove(decomp)

atom_params = [("adf-ch4-freq.t21.ascii.xz", "adf-ch4-atom.csv.xz"),
               ("adf-ethane-ts-freq.t21.ascii.xz", "adf-ethane-atom.csv.xz")]
freq_params = [("adf-ch4-freq.t21.ascii.xz", "adf-ch4-frequency.csv.xz"),
               ("adf-ethane-ts-freq.t21.ascii.xz", "adf-ethane-frequency.csv.xz")]

@pytest.mark.parametrize("res_file,test_file", atom_params)
def test_atom(editor, test_file):
    data = pd.read_csv(resource(test_file), compression='xz', header=0, index_col=False)
    editor.parse_atom()
    cols = ['x', 'y', 'z', 'Z']
    assert np.allclose(data[cols].values, editor.atom[cols].values)

@pytest.mark.parametrize("res_file,test_file", freq_params)
def test_frequency(editor, test_file):
    data = pd.read_csv(resource(test_file), compression='xz', header=0, index_col=False)
    editor.parse_frequency()
    cols = ['dx', 'dy', 'dz', 'frequency']
    assert np.allclose(data[cols].values, editor.frequency[cols].values)

