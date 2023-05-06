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
    delfq_zero = pd.DataFrame(delfq_zero)
    delfq_plus = pd.DataFrame(delfq_plus)
    delfq_minus = pd.DataFrame(delfq_minus)
    return [delfq_zero, delfq_plus, delfq_minus]

def _perform_1d_derivative(plus, minus, coeffs, delta):
    '''
    Calculate the numerical first derivative. This is generalized for
    any finite difference method as long as the prefactors for each of
    the steps is given.

    Note:
        This assumes that the data is ordered such that the first
        element in the coefficients (`coeffs`) aligns with the first
        element in the data arrays (`plus` and `minus`).

    Args:
        plus (:class:`numpy.ndarray`): Array with the values for the
                positive displacement.
        minus (:class:`numpy.ndarray`): Array with the values for the
                negative displacement.
        coeffs (:obj:`list` of list-like): Coefficient prefactors to be
                used in the numerical differentiation. These are the
                scaling factors when performing the different types of
                central finite difference calculations.
        delta (:obj:`float`): Displacment parameter used.

    Returns:
        deriv (:obj:`float`): Numerical derivative of the given data
                sets.
    '''
    deriv = 0
    arrs = np.array([plus, minus]).T
    for idx, arr in enumerate(zip(*arrs)):
        for jdx, coeff in enumerate(coeffs):
            deriv += (-1)**idx*coeff*arr[jdx]
    deriv /= delta
    return deriv

def _check_array_size(arr, size, label):
    msg = "Did not detect the right number of elements in the given " \
          +"array for the {} displacements. Expected {}, but got {}."
    if arr.shape[0] != size:
        raise ValueError(msg.format(label, size, arr.shape[0]))


def two_point_1d(plus, minus, delta):
    '''
    Two point central finite difference method to approximate the first
    derivative of the input data.

    Args:
        plus (:obj:`float`): Data frame with the positive displacement
                data.
        minus (:obj:`float`): Data frame with the negative displacement
                data.
        delta (:obj:`float`): Displacement parameter used.

    Returns:
        deriv (:obj:`float`): Numerical first derivative for given data.
    '''
    deriv = (plus - minus)/(2*delta)
    return deriv

def four_point_1d(plus, minus, delta):
    '''
    Four point central finite difference method to approximate the first
    derivative of the input data, also know as the five point stencil.

    Args:
        plus (:obj:`float`): Data frame with the positive displacement
                data.
        minus (:obj:`float`): Data frame with the negative displacement
                data.
        delta (:obj:`float`): Displacement parameter used.

    Returns:
        deriv (:obj:`float`): Numerical first derivative for given data.
    '''
    # make sure that the input data is of the right size
    _check_array_size(plus, 2, 'positive')
    _check_array_size(minus, 2, 'negative')
    # calculated coefficient prefactors for each displacement
    coeffs = [2./3, -1./12]
    deriv = _perform_1d_derivative(plus, minus, coeffs, delta)
    return deriv

def six_point_1d(plus, minus, delta):
    '''
    Six point central finite difference method to approximate the first
    derivative of the input data.

    Args:
        plus (:obj:`float`): Data frame with the positive displacement
                data.
        minus (:obj:`float`): Data frame with the negative displacement
                data.
        delta (:obj:`float`): Displacement parameter used.

    Returns:
        deriv (:obj:`float`): Numerical first derivative for given data.
    '''
    # make sure that the input data is of the right size
    _check_array_size(plus, 3, 'positive')
    _check_array_size(minus, 3, 'negative')
    # calculated coefficient prefactors for each displacement
    coeffs = [3./4, -3./20, 1./60]
    deriv = _perform_1d_derivative(plus, minus, coeffs, delta)
    return deriv

def eight_point_1d(plus, minus, delta):
    # make sure that the input data is of the right size
    _check_array_size(plus, 4, 'positive')
    _check_array_size(minus, 4, 'negative')
    # calculated coefficient prefactors for each displacement
    coeffs = [4./5, -1./5, 4./105, -1./280]
    deriv = _perform_1d_derivative(plus, minus, coeffs, delta)
    return deriv

def _perform_2d_derivative(plus, minus, equil, coeffs, delta):
    deriv = equil*coeffs[0]
    arrs = np.array([plus, minus]).T
    for arr in zip(*arrs):
        for jdx, coeff in enumerate(coeffs[1:]):
            deriv += coeff*arr[jdx]
    deriv /= delta**2
    return deriv

def two_point_2d(plus, minus, equil, delta):
    coeffs = [-2., 1.]
    deriv = (plus - 2*equil + minus) / delta**2
    return deriv

def four_point_2d(plus, minus, equil, delta):
    # make sure that the input data is of the right size
    _check_array_size(plus, 2, 'positive')
    _check_array_size(minus, 2, 'negative')
    coeffs = [-5./2, 4./3, -1./12]
    deriv = _perform_2d_derivative(plus, minus, equil, coeffs, delta)
    return deriv

def six_point_2d(plus, minus, equil, delta):
    # make sure that the input data is of the right size
    _check_array_size(plus, 3, 'positive')
    _check_array_size(minus, 3, 'negative')
    coeffs = [-49./18, 3./2, -3./20, 1./90]
    deriv = _perform_2d_derivative(plus, minus, equil, coeffs, delta)
    return deriv

def eight_point_2d(plus, minus, equil, delta):
    # make sure that the input data is of the right size
    _check_array_size(plus, 4, 'positive')
    _check_array_size(minus, 4, 'negative')
    coeffs = [-205./72, 8./5, -1./5, 8./315, -1./560]
    deriv = _perform_2d_derivative(plus, minus, equil, coeffs, delta)
    return deriv

def _get_prefac(p, d):
    '''
    Simple function to get the common divisible factor on all the
    displacements. Mainly useful when trying to get the prefactors with
    the root finding algorithm in
    :func:`vibrav.numerical.derivatives._determine_prefactors`.
    '''
    prefacs = np.sum([2*x*y for x, y in zip(p, d)])
    return prefacs

def _determine_prefactors(p, d):
    n = len(p)
    eqs = [2/np.math.factorial(x)*np.sum(p*d**x) \
               for x in range(1, 2*n, 2)]
    eqs[0] -= _get_prefac(p, d)
    return eqs

def _get_arb_coeffs(steps):
    '''
    Calculate the coefficients for the arbitrary displacements.

    Args:
        steps (:class:`numpy.ndarray`): Array with the absolute step
                increments taken. Should be made up of integers.

    Returns:
        coeffs (:class:`numpy.ndarray`): Coefficient prefactors to use
                in the numerical derivative.
    '''
    from scipy.optimize import root
    x0 = np.array([0.5]*steps.shape[0])
    res = root(_determine_prefactors, x0=x0, args=steps)
    prefac = _get_prefac(res.x, steps)
    coeffs = res.x/prefac
    return coeffs

def arb_disps_1d(plus, minus, disp_steps, delta):
    '''
    Unlike the functions
    :func:`vibrav.numerical.derivatives.four_point_1d`,
    :func:`vibrav.numerical.derivatives.six_point_1d`. This function
    serves to calculate a derivative when the displacements are not
    constant.

    Note:
        This requires that the steps are symmetric. Meaning, the
        steps that are taken in the positive direction must be the
        same as in the negative direction.

    Args:
        plus (:class:`numpy.ndarray`): Array with the positive
                displacement data.
        minus (:class:`numpy.ndarray`): Array with the negative
                displacement data.
        disp_steps (:class:`numpy.ndarray`): Array with the displacement
                step size taken.
        delta (:obj:`float`): Displacement taken for first diaplacement
                step.

    Returns:
        deriv (:obj:`float`): Numerical first derivative for given data.
    '''
    # TODO: for now we just assume that the given
    #       data has the right size
    steps = disp_steps/delta
    if not np.allclose(steps, list(map(int, steps))):
        raise ValueError("There is an issue with the given steps taken.")
    coeffs = _get_arb_coeffs(steps)
    deriv = _perform_1d_derivative(plus, minus, coeffs, delta)
    return deriv

