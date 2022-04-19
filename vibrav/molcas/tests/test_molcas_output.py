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
from vibrav import molcas
from vibrav.base import resource
from vibrav.util.io import uncompress_file
import numpy as np
import pandas as pd
import bz2
import lzma
import os
import pytest

@pytest.fixture(scope="module")
def nien_ed():
    nien_decomp = uncompress_file(resource("molcas-rassi-nien.out.xz"), compression='xz')
    nien_ed = molcas.Output(nien_decomp)
    yield nien_ed
    os.remove(nien_decomp)

@pytest.fixture(scope="module")
def niphen_ed():
    niphen_decomp = uncompress_file(resource("molcas-rassi-niphen.out.xz"), compression='xz')
    niphen_ed = molcas.Output(niphen_decomp)
    yield niphen_ed
    os.remove(niphen_decomp)

#@pytest.fixture(params=['nien_ed'])
#def editor(request):
#    return request.getfixturevalue(request.param)

@pytest.mark.parametrize('editor,test',
                         [('nien_ed', resource('molcas-rassi-nien-sf-dipole.csv.xz'))])
def test_sf_dipole(editor, test, request):
    data = pd.read_csv(test, compression='xz', header=0,
                       index_col=False)
    ed = request.getfixturevalue(editor)
    ed.parse_sf_dipole_moment()
    arr = []
    cols = []
    for key, val in ed.sf_dipole_moment.groupby('component'):
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

@pytest.mark.parametrize('editor,test',
                         [('nien_ed', resource('molcas-rassi-nien-sf-angmom.csv.xz'))])
def test_sf_angmom(editor, test, request):
    data = pd.read_csv(test, compression='xz', header=0,
                       index_col=False)
    ed = request.getfixturevalue(editor)
    ed.parse_sf_angmom()
    arr = []
    cols = []
    for key, val in ed.sf_angmom.groupby('component'):
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

@pytest.mark.parametrize('editor,test',
                         [('nien_ed', resource('molcas-rassi-nien-sf-quadrupole.csv.xz'))])
def test_sf_quadrupole(editor, test, request):
    data = pd.read_csv(test, compression='xz',
                       header=0, index_col=False)
    ed = request.getfixturevalue(editor)
    ed.parse_sf_quadrupole_moment()
    arr = []
    cols = []
    for key, val in ed.sf_quadrupole_moment.groupby('component'):
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

@pytest.mark.parametrize('editor,test',
                         [('nien_ed', resource('molcas-rassi-nien-energy.csv.xz'))])
def test_energies(editor, test, request):
    data = pd.read_csv(test, compression='xz', header=0,
                       index_col=False)
    ed = request.getfixturevalue(editor)
    ed.parse_sf_energy()
    ed.parse_so_energy()
    assert np.allclose(data['so'].values, ed.so_energy['energy'].values)
    assert np.allclose(data['sf'].dropna().values, ed.sf_energy['energy'].values)

@pytest.mark.parametrize('editor,test',
                         [('nien_ed', resource('molcas-rassi-nien-oscillators.csv.xz'))])
def test_oscillator(editor, test, request):
    data = pd.read_csv(test, compression='xz', header=0,
                                index_col=False)
    ed = request.getfixturevalue(editor)
    data[['nrow', 'ncol']] -= [1, 1]
    ed.parse_sf_oscillator()
    ed.parse_so_oscillator()
    sf_oscil = data.groupby('theory').get_group('sf')
    sf_oscil = sf_oscil.drop('theory', axis=1).values
    so_oscil = data.groupby('theory').get_group('so')
    so_oscil = so_oscil.drop('theory', axis=1).values
    test_sf = ed.sf_oscillator.copy()
    test_sf[['nrow', 'ncol']] = test_sf[['nrow', 'ncol']].astype(np.uint16)
    test_sf = test_sf.values
    test_so = ed.so_oscillator.copy()
    test_so[['nrow', 'ncol']] = test_so[['nrow', 'ncol']].astype(np.uint16)
    test_so = test_so.values
    assert np.allclose(sf_oscil, test_sf)
    assert np.allclose(so_oscil, test_so)

