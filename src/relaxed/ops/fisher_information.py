from __future__ import annotations

__all__ = (
    "fisher_info",
    "cramer_rao_uncert",
)

from typing import Any

import jax
import jax.numpy as jnp
from chex import Array


def fisher_info(model: Any, pars: Array, data: Array) -> Array:
    return -jax.hessian(lambda p, d: model.logpdf(p, d)[0])(pars, data)


def cramer_rao_uncert(model: Any, pars: Array, data: Array) -> Array:
    inv = jnp.linalg.inv(fisher_info(model, pars, data))
    return jnp.sqrt(jnp.diagonal(inv))
