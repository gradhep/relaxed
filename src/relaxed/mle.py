from __future__ import annotations

__all__ = ("fit", "fixed_poi_fit")

from functools import partial
from typing import TYPE_CHECKING, Any, Callable, cast

import jax
import jax.numpy as jnp
import jaxopt
import optax

from relaxed._types import Array

if TYPE_CHECKING:
    import pyhf


@partial(jax.jit, static_argnames=["objective_fn"])
def _minimize(
    objective_fn: Callable[..., float], init_pars: Array, lr: float, *obj_args: Any
) -> Array:
    converted_fn, aux_pars = jax.closure_convert(objective_fn, init_pars, *obj_args)
    # aux_pars seems to be empty? took that line from jax docs example...
    solver = jaxopt.OptaxSolver(
        fun=converted_fn, opt=optax.adam(lr), implicit_diff=True, maxiter=5000
    )
    return solver.run(init_pars, *obj_args, *aux_pars)[0]


@partial(jax.jit, static_argnames=["model"])
def fit(
    data: Array,
    model: pyhf.Model,
    init_pars: Array,
    lr: float = 1e-3,
) -> Array:
    def fit_objective(pars: Array) -> float:
        return cast(float, -model.logpdf(pars, data)[0])

    fit_res = _minimize(fit_objective, init_pars, lr)
    return fit_res


@partial(jax.jit, static_argnames=["model"])
def fixed_poi_fit(
    data: Array,
    model: pyhf.Model,
    init_pars: Array,
    poi_condition: float,
    lr: float = 1e-3,
) -> Array:
    poi_idx = model.config.poi_index

    def fit_objective(pars: Array) -> float:  # NLL
        """lhood_pars_to_optimize: either all pars, or just nuisance pars"""
        # pyhf.Model.logpdf returns list[float]
        blank = jnp.zeros_like(jnp.asarray(model.config.suggested_init()))
        blank += pars
        return cast(float, -model.logpdf(blank.at[poi_idx].set(poi_condition), data)[0])

    fit_res = _minimize(fit_objective, init_pars, lr)
    blank = jnp.zeros_like(jnp.asarray(model.config.suggested_init()))
    blank += fit_res
    poi_idx = model.config.poi_index
    return blank.at[poi_idx].set(poi_condition)
