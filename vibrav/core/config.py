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
import pandas as pd
import numpy as np
from exatomic.exa.core.numerical import Series

class Config(Series):
    '''
    Base class to read the configuration file given at the start of the different Vibrational
    Averaging modules.

    Global default values.

    +-------------------+---------------------------------------------------+----------------+
    | Attribute         | Description                                       | Default Value  |
    +===================+===================================================+================+
    | delta_file        | Filepath of the delta displacement parameters     | delta.dat      |
    |                   | used to generate the different displaced          |                |
    |                   | structures.                                       |                |
    +-------------------+---------------------------------------------------+----------------+
    | reduced_mass_file | Filepath to the reduced masses of the vibrational | redmass.dat    |
    |                   | modes.                                            |                |
    +-------------------+---------------------------------------------------+----------------+
    | frequency_file    | Filepath of the energies of the normal modes.     | freq.dat       |
    +-------------------+---------------------------------------------------+----------------+
    | smatrix_file      | Filepath of the frequency normal mode             | smatrix.dat    |
    |                   | displacements as a column vector of normalized    |                |
    |                   | cartesian coordinates in atomic units.            |                |
    +-------------------+---------------------------------------------------+----------------+
    | eqcoord_file      | Filepath of the equilibrium coordinates of the    | eqcoord.dat    |
    |                   | molecule. This is given as a column vector of     |                |
    |                   | the coordinates in Angstrom units.                |                |
    +-------------------+---------------------------------------------------+----------------+
    | atom_order_file   | Filepath of the atomic symbols in the correct     | atom_order.dat |
    |                   | order as it appears on the XYZ coordinate file.   |                |
    +-------------------+---------------------------------------------------+----------------+
    | delta_disp        | Constant displacement parameter to generate the   | 0.0            |
    |                   | displaced structures. Only matters for the        |                |
    |                   | `delta_type` parameter for                        |                |
    |                   | :func:`vibrav.util.gen_displaced.gen_delta` is 3. |                |
    +-------------------+---------------------------------------------------+----------------+
    | delta_algorithm   | Delta algorithm used to generate the displaced    | 2              |
    |                   | structures. This is passed as the `delta_type`    |                |
    |                   | parameter in                                      |                |
    |                   | :func:`vibrav.util.gen_displaced.gen_delta`       |                |
    +-------------------+---------------------------------------------------+----------------+
    | delta_value       | Normalization parameter used to generate the      | 0.04           |
    |                   | displaced structures. This is passed as the norm  |                |
    |                   | parameter in                                      |                |
    |                   | :func:`vibrav.util.gen_displaced.gen_delta`       |                |
    +-------------------+---------------------------------------------------+----------------+
    | freqdx            | Set which frequency indices to calculate. This is | -1             |
    |                   | a zero based index and can be set with a '-' for  |                |
    |                   | a range of indices and a ',' or ' ' for specific  |                |
    |                   | indices.                                          |                |
    +-------------------+---------------------------------------------------+----------------+

    '''
    _sname = 'config_file'
    _iname = 'config_elem'
    _default = {'delta_file': ('delta.dat', str),
                'reduced_mass_file': ('redmass.dat', str),
                'frequency_file': ('freq.dat', str),
                'smatrix_file': ('smatrix.dat', str),
                'eqcoord_file': ('eqcoord.dat', str),
                'atom_order_file': ('atom_order.dat', str),
                'delta_disp': (0, float),
                'delta_algorithm': (2, int),
                'delta_value': (0.04, float),
                'freqdx': (-1, int)}

    @classmethod
    def open_config(cls, fp, required, defaults=None):
        '''
        Open and read the config file that is given.
    
        Args:
            fp (:obj:`str`): Filepath to the config file
            required (:obj:`list`): Required arguments that must be present in the config file
            defaults (:obj:`list`, optional): Default arguments for the config file that are
                                                  not necessary. Defaults to :code:`None`
    
        Returns:
            config (:obj:`dict`): Dictionary with all of the elements in the config as keys

        Raises:
            AttributeError: When there is more than one value for a default argument, having more
                            than  one value when the input dictionaries say it should be one value,
                            or when there is a missing required parameter.
            Exception: Default catch when the required parameter is not interpreted correctly and
                       does not fall within any of the coded parameters.
            ValueError: When it cannot set the type that has been given as an input. I.e. converting
                        string character into a number.

        Examples:
            The usage of this module can be as follows.

            >>> from vibrav.base import resource
            >>> config = Config.open_config(resource('molcas-ucl6-2minus-vibronic-config'), required={})
            >>> print(config.to_sring())
            config_elem
            freq_data_file              [ADF_FREQ_FILE]
            number_of_multiplicity                  [2]
            spin_multiplicity                    [3, 1]
            number_of_states                   [42, 49]
            number_of_nuclei                        [7]
            number_of_modes                        [15]
            delta_algorithm                           0
            delta_value                            0.04
            oscillator_spin_states                 [91]
            delta_file                        delta.dat
            reduced_mass_file               redmass.dat
            frequency_file                     freq.dat
            sf_energies_file          [energies-SF.txt]
            so_energies_file             [energies.txt]
            zero_order_file             [ucl-rassi.out]
            delta_disp                                0
            >>> print(type(config.number_of_nuclei), type(config.number_of_nuclei[0]))
            <class 'list'> <class 'str'>
            >>> print(type(config.spin_multiplicity), type(config.spin_multiplicity[0]))
            <class 'list'> <class 'str'>

            Any of the items that are not passed as default or required in the `defaults` or `required`
            parameters respectively will be shown as a list entry as shown in the `number_of_multiplicity`
            entry in the result from the example above.

            Now if we pass some required arguments for `number_of_multiplicity`, `spin_multiplicity`,
            `number_of_states` and `number_of_modes`, as is required in the vibronic coupling calculations
            from :class:`vibrav.vibronic.Vibronic`, the input is now as follows.

            >>> from vibrav.base import resource
            >>> required = {'number_of_multiplicity': int, 'spin_multiplicity': (tuple, int),
            ...             'number_of_states': (tuple, int), 'number_of_nuclei': int}
            >>> config = Config.open_config(resource('molcas-ucl6-2minus-vibronic-config'), required=required)
            >>> print(config.to_sring())
            config_elem
            freq_data_file              [ADF_FREQ_FILE]
            number_of_multiplicity                    2
            spin_multiplicity                    (3, 1)
            number_of_states                   (42, 49)
            number_of_nuclei                          7
            number_of_modes                        [15]
            delta_algorithm                           0
            delta_value                            0.04
            oscillator_spin_states                 [91]
            delta_file                        delta.dat
            reduced_mass_file               redmass.dat
            frequency_file                     freq.dat
            sf_energies_file          [energies-SF.txt]
            so_energies_file             [energies.txt]
            zero_order_file             [ucl-rassi.out]
            delta_disp                                0
            >>> print(type(config.number_of_nuclei))
            <class 'int'>
            >>> print(type(config.spin_multiplicity), type(config.spin_multiplicity[0]))
            <class 'tuple'> <class 'int'>

            Now the required inputs that we gave the function are of the specified types. The
            `required` parameter has to take a dictionary where single values in the input file
            will have a single data type defined in the `required` dictionary. Parameters in the
            input file with more than one input, such as the 'spin_multiplicity' input, will have
            a :obj:`list` or :obj:`tuple` of data types where the first one defined what type of
            list-like object it will be and the second parameter is the data type of the individual
            entries.

            If additional default values are to be passed, the `defaults` parameter must be used. We
            will create a new dummy argument, `test`, that will default to `'Hi I am a string'` as a
            :obj:`str`.

            >>> from vibrav.base import resource
            >>> required = {'number_of_multiplicity': int, 'spin_multiplicity': (tuple, int),
            ...             'number_of_states': (tuple, int), 'number_of_nuclei': int}
            >>> new_default = {'test': ('', str)}
            >>> config = Config.open_config(resource('molcas-ucl6-2minus-vibronic-config'), required=required, defaults=new_default)
            >>> print(config.to_sring())
            config_elem
            freq_data_file              [ADF_FREQ_FILE]
            number_of_multiplicity                    2
            spin_multiplicity                    (3, 1)
            number_of_states                   (42, 49)
            number_of_nuclei                          7
            number_of_modes                        [15]
            delta_algorithm                           0
            delta_value                            0.04
            oscillator_spin_states                 [91]
            delta_file                        delta.dat
            reduced_mass_file               redmass.dat
            frequency_file                     freq.dat
            sf_energies_file          [energies-SF.txt]
            so_energies_file             [energies.txt]
            zero_order_file             [ucl-rassi.out]
            test                       Hi I am a string
            delta_disp                                0
            >>> print(type(config.test))
            <class 'str'>

            We can see that we have added a default argument to the input file that is called `test` with
            a :obj:`str` value of `'Hi I am a string'`. The format of the default parameter that must
            be given is a dictionary where every value is a :obj:`tuple` of 2 objects. The first one
            defines the default value, the second one defines the data type of the default value.

            Any other input that is not defined as a default or required argument is saved in the config
            object as a :obj;`list` of string/s. An example if the `freq_data_file` input in the above
            examples. These are saved in case they are needed later on rather than being deleted as a whole.
        '''
        with open(fp, 'r') as fn:
            # get the lines and replace all newline characters
            lines = list(map(lambda x: x.replace('\n', ''), fn.readlines()))
        # update the defaults dict with the given defaults
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
                    # try to set the type of default value
                    try:
                        config[key] = defaults[key][1](d[1])
                    except ValueError as e:
                        raise ValueError(str(e) \
                                         + ' when reading {} in configuration file'.format(key))
            # check for the required inputs given as a parameter
            # this allows us to use this script in the different ways we need it and we just
            # pass it the parameters that are required for that specific calculation
            # it is not pretty but I think it is the most generalized way to do this
            elif key in required.keys():
                found_required.append(key)
                # make sure if the entry will need to be iterated over
                # this must be specified by giving a two element required param. value in the dict
                if isinstance(required[key], (list, tuple)):
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
            #       we do not want to throw them out as it may be useful at some point
            else:
                config[key] = d[1:]
        # check for missing required arguments
        missing_required = list(filter(lambda x: x not in found_required, required.keys()))
        if missing_required:
            raise AttributeError("There is a required input parameter missing." \
                                +"\nThe required parameters in the config file for this " \
                                +"calculation are:" \
                                +"{}".format('\n - '.join(['']+list(required.keys()))))
        # check for missing default arguments and fill in with the default values
        # availabel in the defaults dict
        missing_default = list(filter(lambda x: x not in found_defaults, defaults.keys()))
        for missing in missing_default:
            config[missing] = defaults[missing][0]
        return cls(config)

