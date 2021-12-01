# relaxed

[![Actions Status][actions-badge]][actions-link]
[![codecov](https://codecov.io/gh/gradhep/relaxed/branch/main/graph/badge.svg?token=CJLGC7H7NY)](https://codecov.io/gh/gradhep/relaxed)
[![Code style: black][black-badge]][black-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]
[![Documentation Status][rtd-badge]][rtd-link]




[actions-badge]:            https://github.com/gradhep/relaxed/workflows/CI/badge.svg
[actions-link]:             https://github.com/gradhep/relaxed/actions
[black-badge]:              https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]:               https://github.com/psf/black
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/relaxed
[conda-link]:               https://github.com/conda-forge/relaxed-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/gradhep/relaxed/discussions
[gitter-badge]:             https://badges.gitter.im/https://github.com/gradhep/relaxed/community.svg
[gitter-link]:              https://gitter.im/https://github.com/gradhep/relaxed/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
[pypi-link]:                https://pypi.org/project/relaxed/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/relaxed
[pypi-version]:             https://badge.fury.io/py/relaxed.svg
[rtd-badge]:                https://readthedocs.org/projects/relaxed/badge/?version=latest
[rtd-link]:                 https://relaxed.readthedocs.io/en/latest/?badge=latest
[sk-badge]:                 https://scikit-hep.org/assets/images/Scikit--HEP-Project-blue.svg


Provides differentiable ("relaxed") versions of common operations in high-energy physics. Where possible, function APIs try to mimic their commonly used counterparts, e.g. fitting and hypothesis testing in [`pyhf`](github.com/scikit-hep/pyhf).

Currently implemented:
- [`relaxed.hist`](src/relaxed/ops/histograms.py): histograms via kernel density estimation
- fitting routines:
  - [`relaxed.mle.fit`](src/relaxed/mle/global_fit.py): global MLE fit
  - [`relaxed.mle.fixed_poi_fit`](src/relaxed/infer/hypothesis_test.py): constrained fit given a value of a parameter of interest
- [`relaxed.infer.hypotest`](src/relaxed/infer/hypothesis_test.py): hypothesis test using the profile likelihood as a test statistic

## install
```
pip install relaxed
```

For use with `pyhf`, e.g. in a [`neos`](github.com/gradhep/neos)-type workflow, it is temporarily recommended to install `pyhf` using a specific branch that is designed to be differentiable with respect to model construction:

```
pip install git+http://github.com/scikit-hep/pyhf.git@make_difffable_model_ctor
```
We plan to merge this into `pyhf` when it's stable, and will then drop this instruction :)
