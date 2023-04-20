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
'''
Base module
###########
Handles the resource files.
'''
import vibrav
import os

def _get_static_path():
    return os.sep.join(vibrav.__file__.split(os.sep)[:-1]+['static'])

def resource(file):
    '''
    Get the requested resource file from the static directory.

    Args:
        file (:obj:`str`): Name of resource file.

    Returns:
        resource_path (:obj:`str`): Absolute path to the resource file.

    Raises:
        ValueError: When there is more than one resource file found with that same name.
        FileNotFoundError: When the resource file cannot be found in the static directory.

    Examples:
        The usage of this function is fairly straightforward where the code below can be
        used to get the filepath of the resource file.

        >>> resource('adf-ch4-atom.csv.xz')
        '/path/to/vibrav/vibrav/static/adf/ch4/adf-ch4-atom.csv.xz'

        To view a full list of resource files please see :func:`vibrav.base.list_resource`
    '''
    abspath, files = list_resource(full_path=True, return_both=True)
    index = []
    for idx, f in enumerate(files):
        if f == file:
            index.append(idx)
    if len(index) > 1:
        raise ValueError("More than one file was found with that name in the static directory. " \
                        +"Please submit an issue on the github page. If this was a file that you " \
                        +"added make sure that it does not have the same name as some of the " \
                        +"existing files, regardless of whether they are in different directories.")
    elif len(index) == 0:
        raise FileNotFoundError("The specified resource file was not found")
    resource_path = abspath[index[0]]
    return resource_path

def list_resource(full_path=False, return_both=False, search_string='', rel_path=False):
    '''
    Get all of the available resource files in the static directory.

    Args:
        full_path (:obj:`bool`, optional): Return the absolute path of the resource files.
                                           Defaults to :code:`False`.
        return_both (:obj:`bool`, optional): Return both the absolute paths and resource files.
                                             Defaults to :code:`False`.
        search_string (:obj:`str`, optional): Regex string to limit the number of entries to return.
                                              Defaults to :code:`''`.
        rel_path (:obj:`bool`, optional): Return the relative path instead of the absolute path to
                                          resource files. Good for testing code. Defaults to
                                          :code:`False`.

    Returns:
        resource_files (:obj:`list`): Resource file list depending on the input parameters.

    Examples:
        The usage of the this function is as follows.

        For a simple list of all the available resource files.

        >>> print(list_resource())
        ['adf-ch4-atom.csv.xz', 'adf-ch4-freq.t21.ascii.xz', 'adf-ch4-frequency.csv.xz',
         'adf-ethane-atom.csv.xz', 'adf-ethane-frequency.csv.xz', 'adf-ethane-ts-freq.t21.ascii.xz',
         'g16-nitromalonamide-freq.out.xz', 'g16-nitromalonamide-zpvc-data.tar.xz',
         'boltz-dist-full-test.csv.xz', 'nien3-1-0.02-delta.dat.xz', 'nien3-1-0.02-freq.dat.xz',
         'nien3-1-0.04-delta.dat.xz', 'nien3-1-0.04-freq.dat.xz', 'nien3-1-0.08-delta.dat.xz',
         'nien3-1-0.08-freq.dat.xz', 'nien3-2-0.02-delta.dat.xz', 'nien3-2-0.02-freq.dat.xz',
         'nien3-2-0.04-delta.dat.xz', 'nien3-2-0.04-freq.dat.xz', 'nien3-2-0.08-delta.dat.xz',
         'nien3-2-0.08-freq.dat.xz', 'nien3-frequency-data.csv.xz', 'nitromalonamide-zpvc-config.conf',
         'nitromalonamide-zpvc-dat-files.tar.xz', 'nitromalonamide-zpvc-geometry.csv.xz',
         'nitromalonamide-zpvc-grad.dat.xz', 'nitromalonamide-zpvc-prop.dat.xz',
         'nitromalonamide-zpvc-results.csv.xz', 'molcas-rassi-nien-degen-so-energy.csv.xz',
         'molcas-rassi-nien-energy.csv.xz', 'molcas-rassi-nien-oscillators.csv.xz',
         'molcas-rassi-nien-sf-angmom.csv.xz', 'molcas-rassi-nien-sf-dipole.csv.xz',
         'molcas-rassi-nien-sf-quadrupole.csv.xz', 'molcas-rassi-nien.out.xz',
         'molcas-ucl6-2minus-eigvectors.txt.xz', 'molcas-ucl6-2minus-energies.txt.xz',
         'molcas-ucl6-2minus-oscillators.txt.xz', 'molcas-ucl6-2minus-sf-dipole-1.txt.xz',
         'molcas-ucl6-2minus-so-dipole-1.txt.xz', 'molcas-ucl6-2minus-vibronic-config',
         'molcas-ucl6-2minus-vibronic-coupling.tar.xz']

        Options have been implemented in this function to make it easier to see what resource
        files there are for a particular type of resource list. Below we show how to get all
        the resource files that are linked to an ADF calculation.

        >>> print(list_resource(search_string='adf'))
        ['adf-ch4-atom.csv.xz', 'adf-ch4-freq.t21.ascii.xz', 'adf-ch4-frequency.csv.xz',
         'adf-ethane-atom.csv.xz', 'adf-ethane-frequency.csv.xz', 'adf-ethane-ts-freq.t21.ascii.xz']

        Similarly, one can view all of the resource files that are used for the ZPVC calculations.

        >>> print(list_resource(search_string='zpvc'))
        ['g16-nitromalonamide-zpvc-data.tar.xz', 'nitromalonamide-zpvc-config.conf',
         'nitromalonamide-zpvc-dat-files.tar.xz', 'nitromalonamide-zpvc-geometry.csv.xz',
         'nitromalonamide-zpvc-grad.dat.xz', 'nitromalonamide-zpvc-prop.dat.xz',
         'nitromalonamide-zpvc-results.csv.xz']

        Or, one can view all of the resource files that are an ascii Tape21 file.

        >>> print(list_resource(search_string='ascii'))
        ['adf-ch4-freq.t21.ascii.xz', 'adf-ethane-ts-freq.t21.ascii.xz']

        Note, these file are in separate directories as shown with the next input parameter
        :code:`full_path` along with the :code:`rel_path` parameter for the sake of the example.
        The command was executed in the path where vibrav was cloned.

        >>> print(list_resource(full_path=True, search_string='ascii', rel_path=True))
        ['vibrav/static/adf/ch4/adf-ch4-freq.t21.ascii.xz',
         'vibrav/static/adf/ethane/adf-ethane-ts-freq.t21.ascii.xz']

        We can also return both the paths to the resource files along with the filenames
        of the resource files.

        >>> print(list_resource(full_path=True, search_string='ascii', rel_path=True))
        [['vibrav/static/adf/ch4/adf-ch4-freq.t21.ascii.xz',
          'vibrav/static/adf/ethane/adf-ethane-ts-freq.t21.ascii.xz'],
         ['adf-ch4-freq.t21.ascii.xz', 'adf-ethane-ts-freq.t21.ascii.xz']]
    '''
    fp = []
    if full_path:
        abspaths = []
    else:
        abspaths = None
    # find and organize all of the files
    for (dirpath, _, files) in os.walk(_get_static_path()):
        # absolute or relative path
        if not rel_path:
            paths = list(map(lambda x: os.path.abspath(os.path.join(dirpath, x)), files))
        else:
            paths = list(map(lambda x: os.path.relpath(os.path.join(dirpath, x)), files))
        for path, file in zip(paths, files):
            if os.path.isfile(path) and search_string in path:
                if not abspaths is None:
                    abspaths.append(path)
                fp.append(file)
    # check input options for what is to be returned
    if return_both:
        resource_files = [abspaths, fp]
    elif full_path and not return_both:
        resource_files = abspaths
    elif not (full_path and return_both):
        resource_files = fp
    else:
        # only raised if all else fails
        # should never actually execute
        raise RuntimeError("Um.....this is embarrasing. " \
                          +"The input for list_resource was not understood")
    return resource_files

