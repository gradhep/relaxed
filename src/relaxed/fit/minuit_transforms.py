"""implementations of scaling transforms to aid minimisation, taken from minuit."""

__all__ = ["to_bounded_vec", "to_bounded", "to_inf_vec", "to_inf"]

import jax.numpy as jnp

from .._types import ArrayDevice


def to_bounded_vec(param: ArrayDevice, bounds: ArrayDevice) -> ArrayDevice:
    """[-inf, inf] -> [bounds[:, 0], bounds[:, 1]]."""
    a, b = bounds[:, 0], bounds[:, 1]
    return a + (b - a) * 0.5 * (jnp.sin(param) + 1.0)


# [-inf, inf] -> [a,b]
def to_bounded(param: ArrayDevice, bounds: ArrayDevice) -> ArrayDevice:
    """[-inf, inf] -> [bounds[0], bounds[1]]."""
    a, b = bounds
    return a + (b - a) * 0.5 * (jnp.sin(param) + 1.0)


# [-inf, inf] <- [a,b] (vectors)
def to_inf_vec(param: ArrayDevice, bounds: ArrayDevice) -> ArrayDevice:
    """[bounds[:, 0], bounds[:, 1]] -> [-inf, inf]."""
    a, b = bounds[:, 0], bounds[:, 1]
    x = 2.0 * (param - a) / (b - a) - 1.0
    return jnp.arcsin(x)


# [-inf, inf] <- [a,b]
def to_inf(param: ArrayDevice, bounds: ArrayDevice) -> ArrayDevice:
    """[bounds[0], bounds[1]] -> [-inf, inf]."""
    a, b = bounds
    x = 2.0 * (param - a) / (b - a) - 1.0
    return jnp.arcsin(x)
