from __future__ import annotations

__all__ = ("asimov_sig", "gaussianity")

from typing import TYPE_CHECKING, Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array, flatten_util
from jax.random import multivariate_normal

from relaxed.ops import fisher_info

if TYPE_CHECKING:
    PyTree = Any
    from jax.typing import ArrayLike


@jax.jit
def asimov_sig(s: Array, b: Array) -> float:
    """Median expected significance for a counting experiment, valid in the asymptotic regime.
    Also valid for the multi-bin case.

    Parameters
    ----------
    s : Array
        Signal counts.
    b : Array
        Background counts.

    Returns
    -------
    float
        The expected significance.
    """
    q0 = 2 * jnp.sum((s + b) * (jnp.log(1 + s / b)) - s)
    return cast(float, q0**0.5)


def _gaussian_logpdf(
    bestfit_pars: Array,
    data: Array,
    cov: Array,
) -> Array:
    return cast(Array, jsp.stats.multivariate_normal.logpdf(data, bestfit_pars, cov))


@eqx.filter_jit
def gaussianity(
    model: PyTree,
    bestfit_pars: dict[str, ArrayLike],
    data: Array,
    rng_key: Any,
    n_samples: int = 1000,
) -> Array:
    # - compare the likelihood of the fitted model with a gaussian approximation
    # that has the same MLE (fitted_pars)
    # - do this across a number of points in parspace (sampled from the gaussian approx)
    # and take the mean squared diff
    # - centre the values wrt the best-fit vals to scale the differences

    cov_approx = jnp.linalg.inv(fisher_info(model, bestfit_pars, data))
    flat_bestfit_pars, tree_structure = flatten_util.ravel_pytree(bestfit_pars)
    gaussian_parspace_samples = multivariate_normal(
        key=rng_key,
        mean=flat_bestfit_pars,
        cov=cov_approx,
        shape=(n_samples,),
    )

    relative_nlls_model = jax.vmap(
        lambda pars, data: -(
            model.logpdf(pars=tree_structure(pars), data=data)
            - model.logpdf(pars=bestfit_pars, data=data)
        ),  # scale origin to bestfit pars
        in_axes=(0, None),
    )(gaussian_parspace_samples, data)

    relative_nlls_gaussian = jax.vmap(
        lambda pars, data: -(
            _gaussian_logpdf(pars, data, cov_approx)
            - _gaussian_logpdf(flat_bestfit_pars, data, cov_approx)
        ),  # data fixes the lhood shape
        in_axes=(0, None),
    )(gaussian_parspace_samples, flat_bestfit_pars)

    diffs = relative_nlls_model - relative_nlls_gaussian
    return jnp.nanmean(diffs**2, axis=0)
