from ._version import version as __version__

__all__ = ("__version__", "hist", "cramer_rao_uncert", "fisher_info", "mle", "infer")

from . import infer, mle
from .ops import cramer_rao_uncert, fisher_info, hist
