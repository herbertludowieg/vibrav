VIBRAtional AVeraging (vibrav)
==============================

A tool to perform vibrational averaging of molecular properties on molecules.

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/e56e338b3e944e1985b846c9127ed952)](https://www.codacy.com/gh/herbertludowieg/vibrav/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=herbertludowieg/vibrav&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/e56e338b3e944e1985b846c9127ed952)](https://www.codacy.com/gh/herbertludowieg/vibrav/dashboard?utm_source=github.com&utm_medium=referral&utm_content=herbertludowieg/vibrav&utm_campaign=Badge_Coverage)

## Installation

To use this code package you must download the development version and execute

`pip intsall -r requirements.txt -e .`

and to build the documantation

`pip install -r requirements.publish.txt -r requirements.txt -e .`

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

## Requirements

- numpy
- pandas
- numba
- [exatomic](https://github.com/exa-analytics/exatomic)

## Calculations available

### Vibronic Coupling:

### Zero-point vibrational corrections

### Vibrational Raman Optical Activity

## Research Publications:

1. Abella, L; Ludowieg, H D; Autschbach, J. Theoretical Study of the Raman Optical Activity Spectra of $\ce{[M(en)3]^{3+}}$ with M = Co, Rh. *Chirality* **2020**, 32, 6, 741 $-$ 752. DOI: [10.1002/chir.23194](https://doi.org/10.1002/chir.23194)
2. Ganguly, G; Ludowieg, H D; Autschbach, J. Ab Initio Study of Vibronic and Magnetic 5f-to-5f and Dipole-Allowed 5f-to-6d and Charge-Transfer Transitions in $\ce{[UX6]^{n−}}$ (X = Cl, Br; n = 1, 2). *J. Chem. Theory Comput.* **2020**, 16, 8, 5189 $-$ 5202. DOI: [10.1021/acs.jctc.0c00386](https://doi.org/10.1021/acs.jctc.0c00386)
3. Atzori, M; Ludowieg, H D; et. al. Validation of microscopic magnetochiral dichroism theory. *Science Advances* **2021**, 7, 17, eabg2859. DOI: [10.1126/sciadv.abg2859](https://doi.org/10.1126/sciadv.abg2859)
4. Morgante, P; Ludowieg, H D; Autschbach, J. Comparative Study of Vibrational Raman Optical Activity with Different Time-Dependent Density Functional Approximations: The VROA36 Database. *J. Phys. Chem. A* **2022**, 126, 9, 2909 $-$ 2927. DOI: [10.1021/acs.jpca.2c00951](https://doi.org/10.1021/acs.jpca.2c00951)

## To cite:

1. Ludowieg, H D. Vibrav: a tool for vibrational averaging. https://github.com/herbertludowieg/vibrav
2. Mort, B C; Autschbach, J. Magnitude of Zero-Point Vibrational Corrections to Optical Rotation in Rigid Organic Molecules:  A Time-Dependent Density Functional Study. *J. Phys. Chem. A* **2005**, 109, 38, 8617 $-$ 8623. DOI: [10.1021/jp051685y](https://doi.org/10.1021/jp051685y)

### Vibronic Coupling

1. Heit, Y N; Gendron, F; Autschbach, J. Calculation of Dipole-Forbidden 5f Absorption Spectra of Uranium(V) Hexa-Halide Complexes. *J. Phys. Chem. Lett.* **2018**, 9, 4, 887 $-$ 894. DOI: [10.1021/acs.jpclett.7b03441](https://doi.org/10.1021/acs.jpclett.7b03441)
2. Ganguly, G; Ludowieg, H D; Autschbach, J. Ab Initio Study of Vibronic and Magnetic 5f-to-5f and Dipole-Allowed 5f-to-6d and Charge-Transfer Transitions in $\ce{[UX6]^{n−}}$ (X = Cl, Br; n = 1, 2). *J. Chem. Theory Comput.* **2020**, 16, 8, 5189 $-$ 5202. DOI: [10.1021/acs.jctc.0c00386](https://doi.org/10.1021/acs.jctc.0c00386)

## Copyright

Copyright (c) 2023, Herbert D Ludowieg

