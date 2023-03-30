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

def get_pos_neg_gradients(grad, freq, nmodes):
    '''
    Here we get the gradients of the equilibrium, positive and negative
    displaced structures.  We extract them from the gradient dataframe
    and convert them into normal coordinates by multiplying them by the
    frequency normal mode displacement values.

    Args:
        grad (:class:`exatomic.gradient.Gradient`): DataFrame containing
                all of the gradient data
        freq (:class:`exatomic.atom.Frquency`): DataFrame containing all
                of the frequency data
        nmodes (int): Number of normal modes in the molecule.

    Returns:
        delfq_zero (pandas.DataFrame): Normal mode converted gradients
                of equilibrium structure
        delfq_plus (pandas.DataFrame): Normal mode converted gradients
                of positive displaced structure
        delfq_minus (pandas.DataFrame): Normal mode converted gradients
                of negative displaced structure
    '''
    grouped = grad.groupby('file')
    # get gradient of the equilibrium coordinates
    grad_0 = grouped.get_group(0)
    # get gradients of the displaced coordinates in the positive direction
    grad_plus = grouped.filter(lambda x: x['file'].drop_duplicates().values in
                                                                    range(1,nmodes+1))
    snmodes = len(grad_plus['file'].drop_duplicates().values)
    # get gradients of the displaced coordinates in the negative direction
    grad_minus = grouped.filter(lambda x: x['file'].drop_duplicates().values in
                                                                    range(nmodes+1, 2*nmodes+1))
    delfq_zero = freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda x:
                                np.sum(np.multiply(grad_0[['fx', 'fy', 'fz']].values, x.values))).values
    # we extend the size of this 1d array as we will perform some matrix summations with the
    # other outputs from this method
    delfq_zero = np.tile(delfq_zero, snmodes).reshape(snmodes, nmodes)
    delfq_plus = grad_plus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                            freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                np.sum(np.multiply(y.values, x.values)))).values
    delfq_minus = grad_minus.groupby('file')[['fx', 'fy', 'fz']].apply(lambda x:
                            freq.groupby('freqdx')[['dx', 'dy', 'dz']].apply(lambda y:
                                np.sum(np.multiply(y.values, x.values)))).values
    return [delfq_zero, delfq_plus, delfq_minus]

