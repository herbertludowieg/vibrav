from vibrav import molcas
from vibrav.base import resource
import numpy as np
import pandas as pd
import bz2
import lzma
import os
import pytest

@pytest.fixture(scope="module")
def editor():
    comp = resource("molcas-rassi-nien.out.xz")
    decomp = comp.split(os.sep)[-1][:-3]
    #with open(decomp, 'wb') as new_file, bz2.BZ2File(comp, 'rb') as file:
    with open(decomp, 'wb') as new_file, lzma.LZMAFile(comp, 'rb') as file:
        for data in iter(lambda : file.read(100 *1024), b''):
            new_file.write(data)
    editor = molcas.Output(decomp)
    yield editor
    os.remove(decomp)

def test_sf_dipole(editor):
    data = pd.read_csv(resource('molcas-rassi-nien-sf-dipole.csv.xz'), compression='xz', header=0,
                       index_col=False)
    print(editor)
    editor.parse_sf_dipole_moment()
    arr = []
    cols = []
    for key, val in editor.sf_dipole_moment.groupby('component'):
        tmp = val.select_dtypes(np.float64).values.T.flatten()
        arr.append(tmp)
        cols.append(key)
    df = pd.DataFrame(np.transpose(arr), columns=cols)
    for col in cols:
        close = np.allclose(df[col].values, data[col].values)
        notnull = np.all(pd.notnull(df[col]))
        if not close:
            raise ValueError("Dipole values were not found to be equal for column {}".format(col))
            assert False
        if not notnull:
            raise ValueError("Null values were found in dipole data for column {}".format(col))
            assert False

def test_sf_angmom(editor):
    data = pd.read_csv(resource('molcas-rassi-nien-sf-angmom.csv.xz'), compression='xz', header=0,
                       index_col=False)
    editor.parse_sf_angmom()
    arr = []
    cols = []
    for key, val in editor.sf_angmom.groupby('component'):
        tmp = val.select_dtypes(np.float64).values.T.flatten()
        arr.append(tmp)
        cols.append(key)
    df = pd.DataFrame(arr).T
    df.columns = cols
    for col in cols:
        close = np.allclose(df[col].values, data[col].values)
        notnull = np.all(pd.notnull(df[col]))
        if not close:
            raise ValueError("Angmom values were not found to be equal for column {}".format(col))
            assert False
        if not notnull:
            raise ValueError("Null values were found in angmom data for column {}".format(col))
            assert False

def test_sf_quadrupole(editor):
    data = pd.read_csv(resource('molcas-rassi-nien-sf-quadrupole.csv.xz'), compression='xz',
                       header=0, index_col=False)
    editor.parse_sf_quadrupole_moment()
    arr = []
    cols = []
    for key, val in editor.sf_quadrupole_moment.groupby('component'):
        tmp = val.select_dtypes(np.float64).values.T.flatten()
        arr.append(tmp)
        cols.append(key)
    df = pd.DataFrame(np.transpose(arr), columns=cols)
    for col in cols:
        close = np.allclose(df[col].values, data[col].values)
        notnull = np.all(pd.notnull(df[col]))
        if not close:
            raise ValueError("Quadrupole values were not found to be equal for column {}".format(col))
            assert False
        if not notnull:
            raise ValueError("Null values were found in quadrupole data for column {}".format(col))
            assert False

def test_energies(editor):
    data = pd.read_csv(resource('molcas-rassi-nien-energy.csv.xz'), compression='xz', header=0,
                       index_col=False)
    editor.parse_sf_energy()
    editor.parse_so_energy()
    assert np.allclose(data['so'].values, editor.so_energy['energy'].values)
    assert np.allclose(data['sf'].dropna().values, editor.sf_energy['energy'].values)

def test_oscillator(editor):
    data = pd.read_csv(resource('molcas-rassi-nien-oscillators.csv.xz'), compression='xz', header=0,
                                index_col=False)
    data[['nrow', 'ncol']] -= [1, 1]
    editor.parse_sf_oscillator()
    editor.parse_so_oscillator()
    sf_oscil = data.groupby('theory').get_group('sf')
    sf_oscil = sf_oscil.drop('theory', axis=1).values
    so_oscil = data.groupby('theory').get_group('so')
    so_oscil = so_oscil.drop('theory', axis=1).values
    test_sf = editor.sf_oscillator.copy()
    test_sf[['nrow', 'ncol']] = test_sf[['nrow', 'ncol']].astype(np.uint16)
    test_sf = test_sf.values
    test_so = editor.so_oscillator.copy()
    test_so[['nrow', 'ncol']] = test_so[['nrow', 'ncol']].astype(np.uint16)
    test_so = test_so.values
    assert np.allclose(sf_oscil, test_sf)
    assert np.allclose(so_oscil, test_so)

