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
from numba import jit, prange, vectorize, float64, complex128
from vibrav.util.math import abs2

@vectorize([float64(float64, float64)])
def compute_oscil_str(absorption, energy):
    '''Multiply the absorptin and energy'''
    return absorption * energy

@jit(nopython=True, parallel=False)
def compute_d_dq_sf(nstates_sf, dham_dq, eq_sf, energies_sf, dprop_dq_sf, tol=1e-5):
    '''
    Compute the spin-free derivative of the chosen property given by the following equation,

    .. math::
        A = \\sum_{k\\neq 1}\\left<\\psi_k^0|\\mu^e|\\psi_2^0\\right>
            \\frac{\\partial\\left<\\psi_1^0|H|\\psi_k^0\\right>  / \\partial Q_p}{E_1^0 - E_k^0}


    .. math::
        B = \\sum_{k\\neq 2}\\left<\\psi_1^0|\\mu^e|\\psi_k^0\\right>
            \\frac{\\partial\\left<\\psi_1^0|H|\\psi_k^0\\right> / \\partial Q_p}{E_1^0 - E_k^0}


    .. math::
        \\frac{\\partial\\mu_{1,2}^{e}\\left(Q\\right)}{\\partial Q_p} = A + B


    Args:
        nstates_sf (:obj:`int`, input): Number of spin free states.
        dham_dq (:obj:`numpy.array`, input): Derivative of the Hamiltonian with respect to the normal
                                             coordinate.
        eq_sf (:obj:`numpy.array`, input): Spin-free values of the property parsed from the equilibrium
                                           geometry.
        energies_sf (:obj:`numpy.array`, input): Spin-free energies parsed from the equilibrium geometry.
        dprop_dq_sf (:obj:`numpy.array`, output): Spin-free derivative of the property of interest.
        tol (:obj:`float`, optional, input): Tolerance value for the energy differences.
                                             Defaults to :code:`1e-5`.
    '''
    for idx in range(nstates_sf):
        for jdx in range(nstates_sf):
            for kdx in range(nstates_sf):
                if not np.abs(energies_sf[idx] - energies_sf[kdx]) < tol:
                    dprop_dq_sf[idx][jdx] += dham_dq[idx][kdx]*eq_sf[kdx][jdx] \
                                                / (energies_sf[idx] - energies_sf[kdx])
                if not np.abs(energies_sf[jdx] - energies_sf[kdx]) < tol:
                    dprop_dq_sf[idx][jdx] += dham_dq[kdx][jdx]*eq_sf[idx][kdx] \
                                                / (energies_sf[jdx] - energies_sf[kdx])

@jit(nopython=True, parallel=False)
def sf_to_so(nstates_sf, nstates, multiplicity, dprop_dq_sf, dprop_dq_so):
    '''
    Extend the spin-free derivative from the number of spin-free states into the number of
    spin-orbit states. It does not change any of the values, rather it will only duplicate each of
    the spin-free values by its respective multiplicity. In principle it can be thought of as
    follows:

    Given a spin-free derivative matrix made up of 2 doublets and 3 singlets

    .. code-block:: python

        [[d_11, d_12, d_13, d_14, d_15],
         [d_21, d_22, d_23, d_24, d_25],
         [d_31, d_32, d_33, d_34, d_35],
         [d_41, d_42, d_43, d_44, d_45],
         [d_51, d_52, d_53, d_54, d_55]]


    Here, d_mn are the individual matrix elements. Now we extend each by its respective multiplicity,
    assuming that states of different multiplicity do not interact.

    .. code-block:: python

        [[d_11,    0, d_12,    0,    0,    0,    0],
         [   0, d_11,    0, d_12,    0,    0,    0],
         [d_21,    0, d_22,    0,    0,    0,    0],
         [   0, d_21,    0, d_22,    0,    0,    0],
         [   0,    0,    0,    0, d_33, d_34, d_35],
         [   0,    0,    0,    0, d_43, d_44, d_45],
         [   0,    0,    0,    0, d_53, d_54, d_55]]


    Now we have a matrix made of the spin-free elements with the dimension of the spin-orbit states.
    Note, this is not the spin-orbit matrix. This is only to prepare the matrix for the complex
    transformation with the eigenvectors gotten from Molcas in the eigectors.txt file. That is taken
    care of by the following equation,

    .. math::
        \\left<\\psi_1^{SO}|\\mu^e|\\psi_2^{SO}\\right> = \\sum_{k,m}U_{k1}^{0*}U_{m2}^{0}
                                    \\left<\\psi_k|\\mu_{1,2}^{e,SF}\\left(Q\\right)|\\psi_m\\right>


    Args:
        nstates_sf (:obj:`int`, input): Number of spin-free states.
        nstates (:obj:`int`, input): Number of spin-orbit states.
        multiplicity (:obj:`numpy.array`, input): 1D array detailing which spin-free states have which
                                                  multiplicity.
        dprop_dq_sf (:obj:`numpy.array`, input): Array of the spin-free derivative elements.
        dprop_dq_so (:obj:`numpy.array`, output): Array of the spin-free derivatives extended to the size of
                                                  spin-orbit states.
    '''
    sti = 0
    for idx in range(nstates_sf):
        for ist in range(multiplicity[idx]):
            stj = 0
            for jdx in range(nstates_sf):
                for jst in range(multiplicity[jdx]):
                    if multiplicity[idx] == multiplicity[jdx] and ist == jst:
                        dprop_dq_so[sti+ist][stj+jst] = dprop_dq_sf[idx][jdx]
                stj += multiplicity[jdx]
        sti += multiplicity[idx]

@jit(nopython=True, parallel=False)
def compute_d_dq(nstates, eigvectors, prop_so, dprop_dq):
    '''
    Perform complex transformation with the eigen vectors to convert the spin-free derivatives to
    spin-orbit derivatives of the property of interest. The equation is as follows,

    .. math::
        \\left<\\psi_1^{SO}|\\mu^e|\\psi_2^{SO}\\right> = \\sum_{k,m}U_{k1}^{0*}U_{m2}^{0}
                                    \\left<\\psi_k|\\mu_{1,2}^{e,SF}\\left(Q\\right)|\\psi_m\\right>


    Args:
        nstates (:obj:`int`, input): Number of spin-orbit states.
        eigvectors (:obj:`numpy.array`, input): Array containing the eigen vectors read from the eigvectors.txt
                                                file produced by Molcas.
        prop_so (:obj:`numpy.array`, input): Extended spin-free derivatives with the proper size. Output from
                                             `vibrav.numerical.vibronic_func.sf_to_so` function.
        dprop_so (:obj:`numpy.array`, output): Spin-orbit derivatives of the property of interest.
    '''
    tmp = np.zeros((nstates, nstates), dtype=np.complex128)
    for idx in range(nstates):
        for jdx in range(nstates):
            for kdx in range(nstates):
                tmp[idx][jdx] += np.conjugate(eigvectors[kdx][idx]) * prop_so[kdx][jdx]
    for idx in range(nstates):
        for jdx in range(nstates):
            for kdx in range(nstates):
                dprop_dq[idx][jdx] += tmp[idx][kdx] * eigvectors[kdx][jdx]

