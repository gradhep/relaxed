from __future__ import annotations

__all__ = ("asimov_sig",)

from typing import cast

import jax.numpy as jnp
from chex import Array
from jax import jit


@jit
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
    n = s + b
    mu_hat = jnp.sum(n - b) / jnp.sum(s)
    q0 = 2 * jnp.sum(n * (jnp.log(1 + (mu_hat * s) / b)) - mu_hat * s)
    return cast(float, q0**0.5)
