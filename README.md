# smooth
(will) Provide differentiable versions of common HEP operations and objectives.

This small example uses a slightly modified version of the machinery in neos. The backend independence is ripped straight from pyhf, and is only implemented for the functions in [`__init__.py`](https://github.com/gradhep/smooth/blob/master/smooth/__init__.py), as we dont yet have fixed point differentiation for the other backends. The numpy backend shouldnt be used, as it doesnt implement autodiff.

Browse or contribute to the [List of Differentiable Operations](list_of_operations.md).

## Developing

Clone and cd into the repo, then in a python venv run
```
python -m pip install -r requirements.txt
```
to install dependencies.

Be sure to read [`CONTRIBUTING.md`](https://github.com/gradhep/smooth/blob/master/CONTRIBUTING.md) before making a PR!
