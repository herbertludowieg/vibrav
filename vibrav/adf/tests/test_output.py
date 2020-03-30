from vibrav import adf
import pytest
import numpy as np

@pytest.fixture(scope="module")
def editor():
    comp = resource("adf-ch4-freq.t21.ascii.xz")
    decomp = comp.split(os.sep)[-1][:-3]
    with open(decomp, 'wb') as new_file, lzma.LZMAFile(comp, 'rb') as file:
        for data in iter(lambda : file.read(100 *1024), b''):
            new_file.write(data)
    editor = adf.Tape21(decomp)
    yield editor
    os.remove(decomp)

def test_atom(editor):
    data = pd.read_csv(resource('adf-ch4-atoms.csv.xz'), compression='xz', header=0,
                       index_col=0)
    editor.parse_atom()
    cols = ['x', 'y', 'z', 'Z', 'isomass']
    assert np.allclose(data[cols].values, editor.atom[cols].values)

def test_frequency(editor):
    pass
