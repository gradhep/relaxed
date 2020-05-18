# smooth
(will) Provide differentiable versions of common HEP operations and objectives.

This small example uses a slightly modified version of the machinery in neos. The backend independence is ripped straight from pyhf, and is only implemented for the functions in __init__.py, as we dont yet have fixed point diff for the other backends. The numpy backend shouldnt be used, as it doesnt implement autodiff.
