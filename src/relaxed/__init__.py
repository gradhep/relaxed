from relaxed._version import version as __version__

__all__ = (
    "__version__",
    "hist",
    "cramer_rao_uncert",
    "fisher_info",
    "mle",
    "infer",
    "metrics",
    "cut",
    "dummy_pyhf",
)

from relaxed import dummy_pyhf, infer, metrics, mle
from relaxed.ops import cramer_rao_uncert, cut, fisher_info, hist
