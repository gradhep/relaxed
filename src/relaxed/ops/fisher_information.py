from __future__ import annotations

__all__ = ["fisher_info", "cramer_rao_uncert"]

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .._types import Array

if TYPE_CHECKING:
    import pyhf


def fisher_info(model: pyhf.Model, pars: Array, data: Array) -> Array:
    return -jax.hessian(model.logpdf)(pars, data)[0]  # since logpdf returns [value]


def cramer_rao_uncert(model: pyhf.Model, pars: Array, data: Array) -> Array:
    inv = jnp.linalg.inv(fisher_info(model, pars, data))
    return jnp.sqrt(jnp.diagonal(inv))
