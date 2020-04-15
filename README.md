VIBRAtional AVeraging (vibrav)
==============================

A tool to perform vibrational averaging of molecular properties on molecules.

## Installation
To use this code package you must download the development version and execute

`pip intsall -e .`

## Building the documentation
The docs are built with the sphinx-apidoc module. It will generate .txt
files with the contents of the docstrings in the python code. We use
the [sphinx_bootstrap_theme](https://github.com/ryan-roemer/sphinx-bootstrap-theme)
for our documentation. To install it run `pip install sphinx_bootstrap_theme`.

To build the documentation,
``` bash
cd docs/
make html
```

To view the built documentation (assuming already in docs directory),
``` bash
cd build/html
xdg-open index.html
```

Or, on the Windows Subsystem Linux,
``` bash
cd build/html
cmd.exe /C start index.html
```

### Requirements
 - numpy
 - pandas
 - numba
 - [exa](https://github.com/exa-analytics/exa)

## Calculations available
### Vibronic Coupling:
This package can calculate the vibronic coupling of electronic transitions. For more information refer 
to reference (1). We currently support the calculation of the following properties:
* electric dipoles
* magnetic dipole
* electric quadrupoles

## Coming Soon!!!
* Zero-point vibrational corrections (zpvc)
* Vibrational Raman Optical activity (vroa)

### Copyright

Copyright (c) 2020, Herbert D Ludowieg

## References:
1. [J. Phys. Chem. Lett. 2018, 9, 4, 887-894](https://doi.org/10.1021/acs.jpclett.7b03441)

