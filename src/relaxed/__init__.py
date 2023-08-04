"""
Copyright (c) 2023 Nathan Simpson. All rights reserved.

relaxed: Differentiable versions of common HEP operations.
"""


from __future__ import annotations

__version__ = "0.4.0"

__all__ = (
    "__version__",
    "hist",
    "cramer_rao_uncert",
    "fisher_info",
    "mle",
    "infer",
    "metrics",
    "cut",
)

from relaxed import infer, metrics, mle
from relaxed.ops import cramer_rao_uncert, cut, fisher_info, hist
