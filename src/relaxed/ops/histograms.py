"""differentiable implementation of the histogram via kernel density estimation."""
from __future__ import annotations

__all__ = ("hist",)

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from chex import Array


@partial(jax.jit, static_argnames=["density", "reflect_infinities"])
def hist(
    events: Array,
    bins: Array,
    bandwidth: float,  # | None = None,
    density: bool = False,
    reflect_infinities: bool = False,
) -> Array:
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
    reflect_infinities: (bool) if true, reflect  under/overflow bins into boundary bins.
    doing so will ensure (normalised) unit total density,
    as kdes have infinite support.

    Returns
    -------
    counts: 1D array of binned counts
    """
    # bandwidth = bandwidth or events.shape[-1] ** -0.25  # Scott's rule

    bins = jnp.array([-jnp.inf, *bins, jnp.inf]) if reflect_infinities else bins

    edge_hi = bins[1:]  # ending bin edges ||<-
    edge_lo = bins[:-1]  # starting bin edges ->||

    # get cumulative counts (area under kde) for each set of bin edges
    cdf_up = jsp.stats.norm.cdf(edge_hi.reshape(-1, 1), loc=events, scale=bandwidth)
    cdf_dn = jsp.stats.norm.cdf(edge_lo.reshape(-1, 1), loc=events, scale=bandwidth)
    # sum kde contributions in each bin
    counts = (cdf_up - cdf_dn).sum(axis=1)

    if density:  # normalize by bin width and counts for total area = 1
        db = jnp.array(jnp.diff(bins), float)  # bin spacing
        counts = counts / db / counts.sum(axis=0)

    if reflect_infinities:
        counts = (
            counts[1:-1]
            + jnp.array([counts[0]] + [0] * (len(counts) - 3))
            + jnp.array([0] * (len(counts) - 3) + [counts[-1]])
        )

    return counts
