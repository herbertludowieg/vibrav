# -*- coding: utf-8 -*-
# Copyright 2019-2020 Herbert D. Ludowieg
# Distributed under the terms of the Apache License 2.0
from exa.core import Editor
import pandas as pd
import numpy as np

class Output(Editor):
    '''
    This output editor is supposed to work for OpenMolcas.
    Currently it is only designed to parse the data required for this script.
    '''
    # TODO: the parsing algorithm is the same so we can simplify this significantly
    def parse_sf_dipole_moment(self):
        # define the search string
        _retdm = "PROPERTY: MLTPL  1"
        _resta = "STATE"
        component_map = {0: 'x', 1: 'y', 2: 'z'}
        found = self.find(_retdm, keys_only=True)
        if not found:
            raise AttributeError("Could not find the TDM in the output")
        #found = self.find(_retdm, _resta, keys_only=True)
        props = np.array(found)[:3]
        stop = props[0] + 5
        while self[stop].strip(): stop += 1
        data_length = stop - props[0] - 5
        tdm = []
        # this is a bit of a mess but since we have three components
        # we can take a nested for loop of three elements without too
        # much of a hit on performance
        for idx, prop in enumerate(props):
            # keep track of columns parsed so far
            counter = 0
            # find where the data blocks are printed
            starts = np.array(self.find(_resta, start=prop, keys_only=True)) + prop + 2
            # data_length should always be the same
            # we could determine it for each data block but that would
            # be a large number of while loops
            stops = starts + data_length
            # hardcoding but should apply in all of our cases
            n = int(np.ceil(data_length/4))
            dfs = []
            # grab all of the data
            for ndx, (start, stop) in enumerate(zip(starts[:n], stops[:n])):
                ncols = len(self[start-2].split())
                df = self.pandas_dataframe(start, stop, ncol=ncols)
                df[0] -= 1
                # set the indexes as they may be different and drop the column
                df.index = df[0]
                df.drop(0, axis=1, inplace=True)
                # set the columns as they should be
                df.columns = list(range(counter, counter+ncols-1))
                dfs.append(df)
                counter += ncols - 1
            # put the component together
            tdm.append(pd.concat(dfs, axis=1))
        # merge the data from all of the components
        stdm = pd.concat(tdm, ignore_index=True)
        stdm['component'] = np.repeat(['x', 'y', 'z'], data_length)
        self.sf_dipole_moment = stdm

    def parse_sf_quadrupole_moment(self):
        _requad = "PROPERTY: MLTPL  2"
        _resta = "STATE"
        component_map = {0: 'xx', 1: 'xy', 2: 'xz', 3: 'yy', 4: 'yz', 5: 'zz'}
        found = self.find(_requad, keys_only=True)
        if not found:
            raise AttributeError("Could not find the Quadrupoles in the output")
        props = np.array(found)[:6]
        stop = props[0] + 5
        while self[stop].strip(): stop += 1
        data_length = stop - props[0] - 5
        qdm = []
        # this is a bit of a mess but since we have three components
        # we can take a nested for loop of three elements without too
        # much of a hit on performance
        for idx, prop in enumerate(props):
            # keep track of columns parsed so far
            counter = 0
            # find where the data blocks are printed
            starts = np.array(self.find(_resta, start=prop, keys_only=True)) + prop + 2
            # data_length should always be the same
            # we could determine it for each data block but that would
            # be a large number of while loops
            stops = starts + data_length
            # hardcoding but should apply in all of our cases
            # there should be a max of 4 columns of data in each of the 'STATE'
            # data blocks so we get the maximum number of hits that there should be
            # assuming a square matrix of data_length x data_length
            n = int(np.ceil(data_length/4))
            dfs = []
            # grab all of the data
            for ndx, (start, stop) in enumerate(zip(starts[:n], stops[:n])):
                ncols = len(self[start-2].split())
                df = self.pandas_dataframe(start, stop, ncol=ncols)
                df[0] -= 1
                # set the indexes as they may be different and drop the column
                df.index = df[0]
                df.drop(0, axis=1, inplace=True)
                # set the columns as they should be
                df.columns = list(range(counter, counter+ncols-1))
                dfs.append(df)
                counter += ncols - 1
            # put the component together
            qdm.append(pd.concat(dfs, axis=1))
            qdm[-1]['component'] = idx
        # merge the data from all of the components
        sqdm = pd.concat(qdm, ignore_index=True)
        sqdm['component'] = sqdm['component'].map(component_map)
        self.sf_quadrupole_moment = sqdm

    def parse_sf_angmom(self):
        _reangm = "PROPERTY: ANGMOM"
        _resta = "STATE"
        component_map = {0: 'x', 1: 'y', 2: 'z'}
        found = self.find(_reangm, keys_only=True)
        if not found:
            raise AttributeError("Could not find the Angular Momentum in the output")
        props = np.array(found)[:3]
        stop = props[0] + 5
        while self[stop].strip(): stop += 1
        data_length = stop - props[0] - 5
        angm = []
        # this is a bit of a mess but since we have three components
        # we can take a nested for loop of three elements without too
        # much of a hit on performance
        for idx, prop in enumerate(props):
            # keep track of columns parsed so far
            counter = 0
            # find where the data blocks are printed
            starts = np.array(self.find(_resta, start=prop, keys_only=True)) + prop + 2
            # data_length should always be the same
            # we could determine it for each data block but that would
            # be a large number of while loops
            stops = starts + data_length
            # hardcoding but should apply in all of our cases
            # there should be a max of 4 columns of data in each of the 'STATE'
            # data blocks so we get the maximum number of hits that there should be
            # assuming a square matrix of data_length x data_length
            n = int(np.ceil(data_length/4))
            dfs = []
            # grab all of the data
            for ndx, (start, stop) in enumerate(zip(starts[:n], stops[:n])):
                ncols = len(self[start-2].split())
                df = self.pandas_dataframe(start, stop, ncol=ncols)
                df[0] -= 1
                # set the indexes as they may be different and drop the column
                df.index = df[0]
                df.drop(0, axis=1, inplace=True)
                # set the columns as they should be
                df.columns = list(range(counter, counter+ncols-1))
                dfs.append(df)
                counter += ncols - 1
            # put the component together
            angm.append(pd.concat(dfs, axis=1))
            angm[-1]['component'] = idx
        # merge the data from all of the components
        sangm = pd.concat(angm, ignore_index=True)
        sangm['component'] = sangm['component'].map(component_map)
        self.sf_angmom = sangm

    def parse_sf_energy(self):
        _reenerg = " RASSI State "
        found = self.find(_reenerg)
        if not found:
            return
        energies = []
        for _, line in found:
            energy = float(line.split()[-1])
            energies.append(energy)
        rel_energy = list(map(lambda x: x - energies[0], energies))
        df = pd.DataFrame.from_dict({'energy': energies, 'rel_energy': rel_energy})
        self.sf_energy = df

    def parse_so_energy(self):
        _reenerg = " SO-RASSI State "
        found = self.find(_reenerg)
        if not found:
            return
        energies = []
        for _, line in found:
            energy = float(line.split()[-1])
            energies.append(energy)
        rel_energy = list(map(lambda x: x - energies[0], energies))
        df = pd.DataFrame.from_dict({'energy': energies, 'rel_energy': rel_energy})
        self.so_energy = df

