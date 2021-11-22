from ._version import version as __version__

__all__ = ("__version__", "hist", "cramer_rao_uncert", "fisher_info")

from .ops import cramer_rao_uncert, fisher_info, hist
