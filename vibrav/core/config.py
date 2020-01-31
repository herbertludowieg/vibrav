# -*- coding: utf-8 -*-
# Copyright 2019-2020 Herbert D. Ludowieg
# Distributed under the terms of the Apache License 2.0
import pandas as pd
import numpy as np
from exa import Series

class Config(Series):
    _sname = 'config_file'
    _iname = 'config_elem'
    _default = {'delta_file': 'delta.dat', 'reduced_mass_file': 'redmass.dat', 'delta_disp': None,
                'frequency_file': 'freq.dat', 'delta_algorithm': None, 'delta_value': 0.04}

    @classmethod
    def open_config(cls, fp, required, defaults=None):
        '''
        Open and read the config file that is given in the generation of the class instance.
    
        Args:
            fp (str): Filepath to the config file
    
        Returns:
            config (dict): Dictionary with all of the elements in the config as keys

        Raises:
            AttributeError: When there is more than one value for a default argument, having more than 
                            one value when the input dictionaries say it should be one value, or when
                            there is a missing required parameter.
            Exception: Default catch when the required parameter is not interpreted correctly and does
                       not fall within any of the coded parameters.
        '''
        with open(fp, 'r') as fn:
            # get the lines and replace all newline characters
            lines = list(map(lambda x: x.replace('\n', ''), fn.readlines()))
        if defaults is not None:
            defaults.update(cls._default)
        else:
            defaults = cls._default
        config = {}
        found_defaults = []
        found_required = []
        for idx, line in enumerate(lines):
            # get the data on the line and deal with whitespace
            if not line.strip():
                continue
            # ignore '#' as comments
            elif line[0] == '#':
                continue
            d = line.split()
            key, val = [d[0].lower(), d[1:]]
            # start by checking if the key is a default value
            if key in defaults.keys():
                found_defaults.append(key)
                # for default values currently they should only have a single entry
                if len(d[1:]) != 1:
                    raise AttributeError("Got the wrong number of entries in {}".format(key))
                else:
                    config[key] = d[1]
            # check for the required inputs given as a parameter
            # this allows us to use this script in the different ways we need it and we just
            # pass it the parameters that are required for that specific calculation
            # it is not pretty but I think it is the most generalized way to do this
            elif key in required.keys():
                found_required.append(key)
                # make sure if the entry will need to be iterated over
                # this must be specified by giving a two element required param. value in the dict
                if isinstance(required[key], (list, tuple)):
                    # TODO: would like to check the type of the dat that is given but may not really
                    #       have a choice in the matter
                    config[key] = tuple(map(lambda x: required[key][1](x), d[1:]))
                # when the requirement value passed is not two elements but the data on the
                # config file has more than one element
                elif not isinstance(required[key], (list, tuple)) and len(d[1:]) > 1:
                    raise AttributeError("Found more than one element in input although it is " \
                                        +"expected to be a single value")
                # for a single element input
                elif len(d[1:]) == 1:
                    config[key] = required[key](d[1])
                # if all else fails
                # this should never execute
                else:
                    raise Exception("Something strange is going on here")
            # all other inputs
            # TODO: make some extras input thing that will take care of these
            # we do not want to throw them out as it may be useful at some point
            else:
                config[key] = d[1:]
        # check for missing required arguments
        missing_required = list(filter(lambda x: x not in found_required, required.keys()))
        if missing_required:
            raise AttributeError("There is a required input parameter missing." \
                                +"\nThe required parameters in the config file for this " \
                                +"calculation are:" \
                                +"{}".format('\n - '.join(['']+list(required.keys()))))
        # check for missing default arguments and fill in the default values
        missing_default = list(filter(lambda x: x not in found_defaults, defaults.keys()))
        for missing in missing_default:
            config[missing] = defaults[missing]
        return cls(config)

