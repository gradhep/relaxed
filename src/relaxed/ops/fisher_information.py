from __future__ import annotations

__all__ = (
    "fisher_info",
    "cramer_rao_uncert",
)

from typing import Callable

import jax
import jax.numpy as jnp
from chex import Array


def fisher_info(
    logpdf: Callable[[Array, Array], float], pars: Array, data: Array
) -> Array:
    return -jax.hessian(logpdf)(pars, data)


def cramer_rao_uncert(
    logpdf: Callable[[Array, Array], float], pars: Array, data: Array
) -> Array:
    inv = jnp.linalg.inv(fisher_info(logpdf, pars, data))
    return jnp.sqrt(jnp.diagonal(inv))
