from exatomic.base import sym2isomass
import numpy as np

def rmass_mwc(data, symbol):
    '''
    Calculate the reduced masses from the mass-weighted normal modes. With
    the equation,

    .. math::
        \\mu_i = \\left(\\sum_k^{3N} \\left(\\frac{l_{MWCk,i}}
                                            {\\sqrt{m_k}}\\right)^2\\right)^{-1}

    Note:
        This assumes that the normal modes have already been placed in the
        :code:`['dx', 'dy', 'dz']` columns.

    Args:
        data (:class:`pandas.DataFrame`): Data frame the has the mass-weighted
                                          normal modes.
        symbol (:obj:`list`): List-like object that has the atomic symbols.

    Returns:
        r_mass (:class:`numpy.ndarray`): Array containing the calculated reduced
                                         masses in Dalton or atomic mass units
                                         not atomic units of mass.

    Examples:
        Usage of this method is as follows,

        >>> import pandas as pd
        >>> from vibrav.base import resource
        >>> res_file = 'adf-ubr-1minus-b3lyp-numeric-norm-modes-mwc.csv.xz'
        >>> freqs = pd.read_csv(resource(res_file), compression='xz')
        >>> symbols = ['U'] + ['Br']*6
        >>> freqs.groupby('freqdx').apply(rmass_mwc, symbols)
        freqdx
        0      78.918337
        1      78.918337
        2      78.918337
        3      78.918337
        4      78.918337
        5      78.918337
        6      93.006642
        7      93.006642
        8      93.006642
        9      78.918337
        10     78.918337
        11     78.918337
        12    111.682084
        13    111.682084
        14    111.682084
        dtype: float64

        It is also possible to use the method without the groupby/apply methods from a
        pandas data frame. However, the pandas data frame that is passed into the
        function **must** have the :code:`['dx', 'dy', 'dz']` columns present. In
        addition the :code:`symbols` parameter must have all of the atomic symbols that
        exists, including ones that are repeated.
    '''
    cols = ['dx', 'dy', 'dz']
    # get the isotopic masses of the unique atoms
    mapper = sym2isomass(symbol)
    # get a list of the isotopic masses for the given symbols
    mass = list(map(mapper.get, symbol))
    # makes it easier to multiply into the normal modes later
    mass = np.repeat(mass, 3).astype(float)
    mass = mass.reshape(data[cols].shape)
    disps = data[cols].values
    # get the reduced masses
    r_mass = np.sum(np.square(disps)/mass)
    r_mass = 1/r_mass
    return r_mass

def rmass_cart(data, symbol):
    '''
    Calculate the reduced masses from the normalized non-mass-weighted cartesian
    normal modes. With the equation,

    .. math::
        \\mu_i = \\left(\\sum_k^{3N} l_{CARTk,i}^2\\right)^{-1}

    Note:
        This assumes that the normal modes have already been placed in the
        :code:`['dx', 'dy', 'dz']` columns.

    Args:
        data (:class:`pandas.DataFrame`): Data frame the has the non-mass-weighted
                                          cartesian normal modes.
        symbol (:obj:`list`): List-like object that has the atomic symbols.

    Returns:
        r_mass (:class:`numpy.ndarray`): Array containing the calculated reduced
                                         masses in Dalton or atomic mass units
                                         not atomic units of mass.

    Examples:
        Usage of this method is as follows,

        >>> import pandas as pd
        >>> from vibrav.base import resource
        >>> res_file = 'adf-ubr-1minus-b3lyp-numeric-norm-modes-cart.csv.xz'
        >>> freqs = pd.read_csv(resource(res_file), compression='xz')
        >>> symbols = ['U'] + ['Br']*6
        >>> freqs.groupby('freqdx').apply(rmass_cart, symbols)
        freqdx
        0      78.918337
        1      78.918337
        2      78.918337
        3      78.918337
        4      78.918337
        5      78.918337
        6      93.006642
        7      93.006642
        8      93.006642
        9      78.918337
        10     78.918337
        11     78.918337
        12    111.682085
        13    111.682085
        14    111.682085
        dtype: float64

        It is also possible to use the method without the groupby/apply methods from a
        pandas data frame. However, the pandas data frame that is passed into the
        function **must** have the :code:`['dx', 'dy', 'dz']` columns present. In
        addition the :code:`symbols` parameter must have all of the atomic symbols that
        exists, including ones that are repeated.
    '''
    cols = ['dx', 'dy', 'dz']
    # get the isotopic masses of the unique atoms
    mapper = sym2isomass(symbol)
    # get a list of the isotopic masses for the given symbols
    mass = list(map(mapper.get, symbol))
    # makes it easier to multiply into the normal modes later
    mass = np.repeat(mass, 3).astype(float)
    mass = mass.reshape(data[cols].shape)
    disps = data[cols].values
    # get the normalization constants
    norms = np.linalg.norm(disps*np.sqrt(mass))
    # unormalize the normal modes
    disps /= norms
    # get the reduced masses
    r_mass = np.sum(np.square(disps))
    r_mass = 1/r_mass
    return r_mass

