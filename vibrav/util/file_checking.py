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
import numpy as np

def _check_file_continuity(df, prop, nmodes, col_name='file'):
    '''
    Verifies that we have all of the displacements included in
    the dataframe. This is necessary as we require that both the
    positive and negative are present. If one is missing we will
    ignore that normal mode index.

    Args:
        df (pandas.DataFrame): Data frame with all of the pertinent
                               data. Must have a 'file' column.
        prop (str): Name of the property for printing the message
                    and better knowing what went wrong.
        nmodes (int): Number of normal modes in the molecule.

    Returns:
        rdf (pandas.DataFrame): Data frame that contains all of the
                                data that was determined to have the
                                positive and negative displacement
                                pair.
    '''
    # grab the file indeces
    files = df[col_name].drop_duplicates()
    # sort the files by what we know to be the positive and
    # negative displacements
    pos_file = files[files.isin(range(1,nmodes+1))]
    neg_file = files[files.isin(range(nmodes+1, 2*nmodes+1))]-nmodes
    # determine where the values are the same
    intersect = np.intersect1d(pos_file.values, neg_file.values)
    # determine which files indeces are missing from the intersection
    diff = np.unique(np.concatenate((np.setdiff1d(pos_file.values, intersect),
                                     np.setdiff1d(neg_file.values, intersect)), axis=None))
    rdf = df.copy()
    # if there is a difference then we will grab only those indeces
    # for which there is a positive and negative displacement pair
    if len(diff) > 0:
        print("Seems that we are missing one of the {} outputs for frequency {} ".format(prop, diff)+ \
              "we will ignore the {} data for these frequencies.".format(prop))
        rdf = rdf[~rdf[col_name].isin(diff)]
        rdf = rdf[~rdf[col_name].isin(diff+nmodes)]
    return rdf

