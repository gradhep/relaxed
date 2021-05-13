"""differentiable implementations of histograms."""

__all__ = ["hist_kde"]

from typing import Optional
import jax
import jax.numpy as jnp
import jax.scipy as jsc

from .._types import ArrayDevice


@jax.jit
def hist_kde(
    events: ArrayDevice,
    bins: ArrayDevice,
    bandwidth: Optional[float] = None,
    density: bool = False,
) -> ArrayDevice:
    """
    Differentiable implementation of a histogram using kernel density estimation.

    Parameters
    ----------
    events: (jax array-like) 1D data!
    bins: (jax array-like) intervals to calculate counts.
    bandwidth: (float) value that specifies the width of the individual
        distributions (kernels), whose cdfs are averaged over each bin. Defaults
        to Scott's rule -- the same as scipy's.
    density: (bool) whether or not to normalize the histogram to unit area.

    Returns
    -------
    counts: 1D array of binned counts
    """
    bandwidth = bandwidth or events.shape[-1] ** -0.25  # Scott's rule

    edge_hi = bins[1:]  # ending bin edges ||<-
    edge_lo = bins[:-1]  # starting bin edges ->||

    # get cumulative counts (area under kde) for each set of bin edges
    cdf_up = jsc.stats.norm.cdf(edge_hi.reshape(-1, 1), loc=events, scale=bandwidth)
    cdf_dn = jsc.stats.norm.cdf(edge_lo.reshape(-1, 1), loc=events, scale=bandwidth)
    # sum kde contributions in each bin
    counts = (cdf_up - cdf_dn).sum(axis=1)

    if density:  # normalize by bin width and counts for total area = 1
        db = jnp.array(jnp.diff(bins), float)  # bin spacing
        return counts / db / counts.sum(axis=0)

    return counts
