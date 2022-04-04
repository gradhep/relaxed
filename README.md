<p align="center">
  :sleeping:<br>
  <strong>relaxed</strong><br>
  <br>
  <a href="https://github.com/gradhep/relaxed/actions">
    <img alt="GitHub Workflow Status" src="https://github.com/gradhep/relaxed/workflows/CI/badge.svg">
  </a>
  <a href="https://codecov.io/gh/gradhep/relaxed">
    <img alt="Read the Docs" src="https://codecov.io/gh/gradhep/relaxed/branch/main/graph/badge.svg?token=CJLGC7H7NY">
  </a>
  <a href="https://zenodo.org/badge/latestdoi/264991846">
    <img alt="Zenodo DOI" src="https://zenodo.org/badge/264991846.svg">
  </a>
</p>


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


Provides differentiable ("relaxed") versions of common operations in high-energy physics.

Based on [`jax`](http://github.com/google/jax). Where possible, function APIs try to mimic their commonly used counterparts, e.g. fitting and hypothesis testing in [`pyhf`](http://github.com/scikit-hep/pyhf).

## currently implemented:
- **basic operations**:
  - [`relaxed.hist`](src/relaxed/ops/histograms.py): histograms via kernel density estimation (tunable bandwidth).
  - [`relaxed.cut`](src/relaxed/ops/cuts.py): approximates a hard cut with a sigmoid function (tunable slope).
- **fitting routines**:
  - [`relaxed.mle.fit`](src/relaxed/mle/global_fit.py): global MLE fit.
  - [`relaxed.mle.fixed_poi_fit`](src/relaxed/infer/hypothesis_test.py): constrained fit given a value of a parameter of interest.
- **inference**:
  - [`relaxed.infer.hypotest`](src/relaxed/infer/hypothesis_test.py): hypothesis test based on the profile likelihood. Supports test statistics for both limit setting (`q`) and discovery (`q_0`).
  - [`relaxed.fisher_info`](src/relaxed/ops/fisher_information.py): the fisher information matrix (of a `pyhf`-type model).
  - [`relaxed.cramer_rao_uncert`](src/relaxed/ops/fisher_information.py): inverts the fisher information matrix to provide uncertainties valid through the [Cramér-Rao bound](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound).
- **metrics**:
  - [`relaxed.metrics.gaussianity`](src/relaxed/metrics/likelihood_gaussianity.py): an experimental metric that quantifies the mean-squared difference of a likelihood function with respect to its gaussian approximation (covariance calculated using the Cramér-Rao bound above).
  - [`relaxed.metrics.asimov_sig`](src/relaxed/metrics/significance.py): easy access to the (single- and multi-bin) stat-only expected significance.

We're maintaining a list of desired differentiable operations in [`list_of_operations.md`](list_of_operations.md) (thanks to [@cranmer](http://github.com/cranmer)) -- feel free to take inspiration or contribute with a PR if there's one you can handle :)

## install
```
python3 -m pip install relaxed
```

For use with `pyhf`, e.g. in a [`neos`](http://github.com/gradhep/neos)-type workflow, it is temporarily recommended to install `pyhf` using a specific branch that is designed to be differentiable with respect to model construction:

```
python3 -m pip install git+http://github.com/scikit-hep/pyhf.git@make_difffable_model_ctor
```
We plan to merge this into `pyhf` when it's stable, and will then drop this instruction :)

## cite
If you use `relaxed`, please cite us! You should be able to do that from the github UI (top-right, under 'cite this repository'), but if not, see our [Zenodo DOI](https://zenodo.org/badge/latestdoi/264991846) or our [`CITATION.cff`](CITATION.cff).

## acknowledgments
Big thanks to all the developers of the main packages we use (`jax`, `pyhf`, `jaxopt`).
Thanks also to [@dfm](github.com/user/dfm) for the README header inspiration ;)
