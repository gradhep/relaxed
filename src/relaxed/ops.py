from __future__ import annotations

__all__ = ("cut", "hist", "fisher_info", "cramer_rao_uncert")

from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, flatten_util


@partial(jax.jit, static_argnames=["keep"])
def cut(data: Array, cut_val: float, slope: float = 1.0, keep: str = "above") -> Array:
    """Use a sigmoid function as an approximate cut. Same as a hard cut in the limit of infinite slope.
    Note: this function returns weights, not indices.

    Parameters
    ----------
    data : Array
        The data to cut.
    cut_val : float
        The value to cut at.
    slope : float
        The slope of the sigmoid function.
    keep : str, optional
        Whether to keep the data above or below the cut. One of:
        - "above" (default)
        - "below"

    Returns
    -------
    Array
        Weighted yields of the data after the cut.
    """
    if keep == "above":
        return 1 / (1 + jnp.exp(-slope * (data - cut_val)))
    if keep == "below":
        return 1 / (1 + jnp.exp(slope * (data - cut_val)))
    msg = f"keep must be one of 'above' or 'below', not {keep}"
    raise ValueError(msg)


@partial(jax.jit, static_argnames=["density", "reflect_infinities"])
def hist(
    data: Array,
    bins: Array,
    bandwidth: float,  # | None = None,
    weights: Array | None = None,
    density: bool = False,
    reflect_infinities: bool = False,
) -> Array:
    """Differentiable histogram, defined via a binned kernel density estimate (bKDE).

    Parameters
    ----------
    data : Array
        1D array of data to histogram.
    bins : Array
        1D array of bin edges.
    bandwidth : float
        The bandwidth of the kernel. Bigger == lower gradient variance, but more bias.
    density : bool
        Normalise the histogram to unit area.
    reflect_infinities : bool
        If True, define bins at +/- infinity, and reflect their mass into the edge bins.

    Returns
    -------
    Array
        1D array of bKDE counts.
    """
    # bandwidth = bandwidth or events.shape[-1] ** -0.25  # Scott's rule

    bins = jnp.array([-jnp.inf, *bins, jnp.inf]) if reflect_infinities else bins

    # get cumulative counts (area under kde) for each set of bin edges
    # account for data with non-integer weights
    if weights is None:
        weights = jnp.ones_like(data)
    cdf = jsp.stats.norm.cdf(bins.reshape(-1, 1), loc=data, scale=bandwidth) * weights
    # sum kde contributions in each bin
    counts = (cdf[1:, :] - cdf[:-1, :]).sum(axis=1)

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


@jax.jit
def fisher_info(model: Any, pars: dict[str, Array], data: Array) -> Array:
    """Fisher information matrix for a model with a logpdf method.

    Parameters
    ----------
    model : Any
        The model to compute the Fisher information matrix for.
        Needs to have a logpdf method (that returns list[float] for now).
    pars : dict[str, Array]
        The (MLE) parameters of the model, as a dict of arrays/floats.
    data : Array
        The data to compute the Fisher information matrix for.

    Returns
    -------
    Array
        Fisher information matrix of shape (num_pars, num_pars).
        Order of columns is the same as the order of the parameters in pars.
        Parameters with multiple dimensions are flattened into their own columns.
    """

    flat_pars, tree_structure = flatten_util.ravel_pytree(pars)

    def lpdf(pars, data):  # handle keyword arguments
        return model.logpdf(pars=tree_structure(pars), data=data)

    return jnp.linalg.inv(-jax.hessian(lpdf)(flat_pars, data))


@eqx.filter_jit
def cramer_rao_uncert(
    model: Any, pars: dict[str, Array], data: Array, return_tree=True
) -> Array:
    """Approximate uncertainties on MLE parameters for a model with a logpdf method.
    Defined as the square root of the diagonal of the Fisher information matrix, valid
    via the Cramer-Rao lower bound.

    Parameters
    ----------
    model : Any
        The model to compute the Cramer-Rao uncertainty for.
        Needs to have a logpdf method (that returns list[float] for now).
    pars : Array
        The (MLE) parameters of the model.
    data : Array
        The data to compute the uncertainty for.

    Returns
    -------
    Array
        Cramer-Rao uncertainty on the MLE parameters.
    """
    fisher = fisher_info(model, pars, data)
    uncert = jnp.sqrt(jnp.diag(fisher))
    if return_tree:
        _, tree_structure = flatten_util.ravel_pytree(pars)
        return tree_structure(uncert)
    return uncert
