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
from exatomic.util.conversions import Ha2inv_cm
import numpy as np
import warnings

def numerical_frequencies(delfq_plus, delfq_minus, redmass,
                          nmodes, delta):
    '''
    Here we calculated the frequencies from the gradients calculated
    for each of the displaced structures along the normal mode. In
    principle this should give the same or nearly the same frequency
    value as that from a frequency calculation.

    Args:
        delfq_0 (numpy.ndarray): Array that holds all of the
            information about the gradient derivative of the
            equlilibrium coordinates.
        delfq_plus (numpy.ndarray): Array that holds all of the
            information about the gradient derivative of the
            positive displaced coordinates.
        delfq_minus (numpy.ndarray): Array that holds all of the
            information about the gradient derivative of the
            negative displaced coordinates.
        redmass (numpy.ndarray): Array that holds all of the
            reduced masses. We can handle both a subset of the
            entire values or all of the values.
        select_freq (numpy.ndarray): Array that holds the selected
            frequency indexes.
        delta (numpy.ndarray): Array that has the delta values used
            in the displaced structures.

    Returns:
        frequencies (numpy.ndarray): Frequency array from the
            calculation.
    '''
    # calculate force constants
    kqi = np.zeros(nmodes)
    #print(redmass_sel.shape)
    for fdx in range(nmodes):
        kqi[fdx] = (delfq_plus[fdx][fdx] - delfq_minus[fdx][fdx]) / (2.0*delta[fdx])
    vqi = np.divide(kqi, redmass.reshape(nmodes,))
    # TODO: Check if we want to exit the program if we get a negative force constant
    n_force_warn = vqi[vqi < 0.]
    if n_force_warn.any() == True:
        # TODO: point to exactly which frequencies are negative
        negative = np.where(vqi<0)[0]
        text = ''
        # frequencies are base 0
        for n in negative[:-1]: text += str(n)+', '
        text += str(negative[-1])
        warnings.warn("Negative force constants have been calculated for frequencies " \
                      +"{} be wary of results".format(text),
                      Warning)
    # return calculated frequencies
    frequencies = np.sqrt(vqi).reshape(nmodes,)*Ha2inv_cm
    return frequencies

