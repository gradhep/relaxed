from __future__ import annotations

__all__ = ("fixed_poi_fit",)

from functools import partial
from typing import TYPE_CHECKING, Callable, cast

import jax
import jax.numpy as jnp
from chex import Array

if TYPE_CHECKING:
    import pyhf

from .minimize import _minimize


def fixed_poi_fit_objective(
    data: Array,
    model: pyhf.Model,
) -> Callable[[Array, float], float]:
    poi_idx = model.config.poi_index

    def fit_objective(
        lhood_pars_to_optimize: Array, poi_condition: float
    ) -> float:  # NLL
        """lhood_pars_to_optimize: either all pars, or just nuisance pars"""
        # pyhf.Model.logpdf returns list[float]
        blank = jnp.zeros_like(jnp.asarray(model.config.suggested_init()))
        blank += lhood_pars_to_optimize
        return cast(float, -model.logpdf(blank.at[poi_idx].set(poi_condition), data)[0])

    return fit_objective


@partial(jax.jit, static_argnames=["model"])  # forward pass
def fixed_poi_fit(
    data: Array,
    model: pyhf.Model,
    init_pars: Array,
    poi_condition: float,
    lr: float = 1e-2,  # arbitrary
) -> Array:
    obj = fixed_poi_fit_objective(data, model)
    fit_res = _minimize(obj, init_pars, lr, poi_condition)
    blank = jnp.zeros_like(jnp.asarray(model.config.suggested_init()))
    blank += fit_res
    poi_idx = model.config.poi_index
    return blank.at[poi_idx].set(poi_condition)
