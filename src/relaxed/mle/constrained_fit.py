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
    data: Array, model: pyhf.Model, poi_condition: float
) -> Callable[[Array], float]:
    def fit_objective(lhood_pars_to_optimize: Array) -> float:  # NLL
        """lhood_pars_to_optimize: either all pars, or just nuisance pars"""
        poi_idx = model.config.poi_index
        pars = lhood_pars_to_optimize.at[poi_idx].set(poi_condition)
        # pyhf.Model.logpdf returns list[float]
        return cast(float, -model.logpdf(pars, data)[0])

    return fit_objective


@partial(jax.jit, static_argnames=["model"])  # forward pass
def fixed_poi_fit(
    data: Array,
    model: pyhf.Model,
    init_pars: Array,
    poi_condition: float,
    lr: float = 1e-3,  # arbitrary
) -> Array:
    obj = fixed_poi_fit_objective(data, model, poi_condition)
    fit_res = _minimize(obj, init_pars, lr)
    blank = jnp.zeros_like(init_pars)
    blank += fit_res
    poi_idx = model.config.poi_index
    return blank.at[poi_idx].set(poi_condition)
