vibrav
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/vibrav.svg?branch=master)](https://travis-ci.com/REPLACE_WITH_OWNER_ACCOUNT/vibrav)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/REPLACE_WITH_OWNER_ACCOUNT/vibrav/branch/master)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/vibrav/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/vibrav/branch/master)

A tool to perform vibrational averaging of molecular properties on molecules.

## Installation
To use this code package you must download the development version and execute

`pip intsall -e .`

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

This program is distributed under the [LGPLv3](https://www.gnu.org/licenses/lgpl-3.0.en.html)
open-source license

## References:
1. [J. Phys. Chem. Lett. 2018, 9, 4, 887-894](https://doi.org/10.1021/acs.jpclett.7b03441)

