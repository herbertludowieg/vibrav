from vibrav.core import Config
from vibrav.base import resource
import pandas as np

def test_config():
    required = {'number_of_multiplicity': int, 'spin_multiplicity': (tuple, int),
                 'number_of_states': (tuple, int), 'number_of_nuclei': int,
                 'number_of_modes': int, 'zero_order_file': str,
                 'oscillator_spin_states': int}
    default= {'sf_energies_file': ('', str), 'so_energies_file': ('', str),
              'angmom_file': ('angmom', str), 'dipole_file': ('dipole', str),
              'spin_file': ('spin', str), 'quadrupole_file': ('quadrupole', str),
              'degen_delta': (1e-5, float)}
    config = Config.open_config(resource('molcas-ucl6-2minus-vibronic-config'), required=required,
                                defaults=default)
    base = {'number_of_multiplicity': 2, 'spin_multiplicity': (3, 1), 'number_of_states': (42, 49),
            'number_of_nuclei': 7, 'number_of_modes': 15, 
            'oscillator_spin_states': 91, 'delta_file': 'delta.dat',
            'reduced_mass_file': 'redmass.dat', 'frequency_file': 'freq.dat',
            'sf_energies_file': 'energies-SF.txt', 'so_energies_file': 'energies.txt',
            'zero_order_file': 'ucl-rassi.out', 'degen_delta': 1e-5}
    for key, val in base.items():
        assert val == config[key]


