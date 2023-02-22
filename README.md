VIBRAtional AVeraging (vibrav)
==============================

A tool to perform vibrational averaging of molecular properties on molecules.

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e56e338b3e944e1985b846c9127ed952)](https://www.codacy.com/gh/herbertludowieg/vibrav/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=herbertludowieg/vibrav&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/e56e338b3e944e1985b846c9127ed952)](https://www.codacy.com/gh/herbertludowieg/vibrav/dashboard?utm_source=github.com&utm_medium=referral&utm_content=herbertludowieg/vibrav&utm_campaign=Badge_Coverage)

## Installation

To use this code package you must download the development version and execute

`pip intsall -e .`

## Building the documentation

The docs are built with the sphinx-apidoc module. It will generate .txt
files with the contents of the docstrings in the python code. We use
the [sphinx_bootstrap_theme](https://github.com/ryan-roemer/sphinx-bootstrap-theme)
for our documentation. To install it run `pip install sphinx_bootstrap_theme`.

To build the documentation,

```bash
cd docs/
make html
```

To view the built documentation (assuming already in docs directory),

```bash
cd build/html
xdg-open index.html
```

Or, on the Windows Subsystem Linux,

```bash
cd build/html
cmd.exe /C start index.html
```

### Requirements

- numpy
- pandas
- numba
- [exatomic](https://github.com/exa-analytics/exatomic)

## Calculations available

### Vibronic Coupling:

This package can calculate the vibronic coupling of electronic transitions. For more information refer 
to reference (1). We currently support the calculation of the following properties:

* electric dipoles
* magnetic dipole
* electric quadrupoles

### Zero-point vibrational corrections

### Vibrational Raman Optical Activity

### Copyright

## Research Publications:

1. Abella, L; Ludowieg, H D; Autschbach, J. Theoretical Study of the Raman Optical Activity Spectra of $\ce{[M(en)3]^{3+}}$ with M = Co, Rh. *Chirality* **2020**, 32 (6), 741 $-$ 752. DOI: [10.1002/chir.23194](https://doi.org/10.1002/chir.23194)

Copyright (c) 2023, Herbert D Ludowieg

## References:

1. [J. Phys. Chem. Lett. 2018, 9, 4, 887-894](https://doi.org/10.1021/acs.jpclett.7b03441)
