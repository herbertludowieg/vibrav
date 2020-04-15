import numpy as np
import pandas as pd
import warnings
from exa.util.constants import Boltzmann_constant as boltz_constant
from exa.util.units import Energy

def boltz_dist(energies, temp, tol=1e-6, states=None):
    '''
    Generate the vibrational boltzmann distributions for all vibrational
    frequencies up to when the distribution is less than the given tolerance
    value.

    Args:
        energies (:obj:`numpy.array`): Array of energies in wavenumbers.
        temp (:obj:`float`): Temperature to calculate the distribution.
        tol (:obj:`float`, optional): Tolerance threshold to cutoff the calculation
                                      of the distribution factors. Defaults to `1e-5`.
        states (:obj:`int`, optional): Number of states to calculate. Defaults to
                                       `None` (all states with a distribution
                                       less than the `boltz_tol` value for the lowest 
                                       frequency).

    Returns:
        boltz (pd.DataFrame): Data frame of the boltzmann factors for each energy.

    Raises:
        ValueError: If the `states` parameter is zero.
        ValueError: If the `temp` paramter is less than or equal to zero.
        ValueError: If the program calculates more than 1000 states at a given tolerance.
    '''
    # to simplify code
    def _boltzmann(energ, nu, temp):
        ''' Boltzmann distribution function. '''
        return np.exp(-energ*nu/(boltz_constant*Energy['J', 'cm^-1']*temp))
    # resursive function
    def _partition_func(energ, nu, temp):
        ''' Partition function. '''
        Q = 0
        if nu == 0:
            # terminating condition
            return _boltzmann(energ, nu, temp)
        else:
            # recursive portion
            Q += _boltzmann(energ, nu, temp) + _partition_func(energ, nu-1, temp)
        return Q
    # for single energy values
    # not the default use of this script
    if isinstance(energies, float): energies = [energies]
    def_states = False
    # toss warning if states is not none
    if states is not None:
        warnings.warn("Calculating only the first {} ".format(states) \
                      +"states for the Boltzmann distribution.", Warning)
        max_nu = states
    elif states == 0:
        raise ValueError("The states parameter given was found to be zero. " \
                         "This cannot be understood and a non-sensical value.")
    else:
        # set to a high value that should never be rached
        # can serve as a final failsafe for the while loop
        states = 1e4
        def_states = True
        # ensure that at least two states are calculated
        max_nu = 2
    # make sure the temp is 'real'
    if temp <= 0:
        raise ValueError("A negative or zero temperature was detected. The input temperature " \
                         +"value must be in units of Kelvin and be a non-zero positive value.")
    # reorder the energies in increasing order
    sorted_energies = pd.Series(energies).sort_values()
    boltz_factors = []
    partition = []
    for idx, (fdx, freq) in enumerate(sorted_energies.items()):
        boltz_factors.append([])
        nu = 0
        # using a while loop as it is easier to define multiple
        # termination conditions
        if def_states:
            while (_boltzmann(freq, nu, temp) > tol or nu < max_nu) and nu < 1e4:
                boltz_factors[-1].append(_boltzmann(freq, nu, temp))
                nu += 1
        else:
            while nu < max_nu:
                boltz_factors[-1].append(_boltzmann(freq, nu, temp))
                nu += 1
        if nu >= 1e4:
            raise ValueError("There is something wrong with this frequency ({}) and ".format(freq) \
                             +"tolerance ({}) combination ".format(tol) \
                             +"as we have calculated more than 1e4 states.")
        # set the maximum number of states to be calculated based on
        # the smallest frequency value to have all same size lists
        if idx == 0 and def_states: max_nu = nu
        # calculate the partition function
        # we subtract 1 from nu because of the termination condition in the while loop
        q = _partition_func(freq, nu-1, temp)
        boltz_factors[-1] /= q
        partition.append(q)
    # put all of the boltzmann factors together
    data = np.stack(boltz_factors, axis=0)
    # make a dataframe for easier data handling
    boltz = pd.DataFrame(data)
    # append some values to keep track of things
    boltz['freqdx'] = sorted_energies.index
    boltz['partition'] = partition
    return boltz

