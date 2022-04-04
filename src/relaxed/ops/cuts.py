from functools import partial

import jax.numpy as jnp
from chex import Array
from jax import jit


@partial(jit, static_argnames=["keep"])
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
    elif keep == "below":
        return 1 / (1 + jnp.exp(slope * (data - cut_val)))
    else:
        raise ValueError(f"keep must be one of 'above' or 'below', not {keep}")
