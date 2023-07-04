from __future__ import annotations

__all__ = ("fit", "fixed_poi_fit")

import inspect
from typing import TYPE_CHECKING, Any, Callable, cast

import jax.numpy as jnp
import jaxopt
from equinox import filter_jit

if TYPE_CHECKING:
    from jax import Array

    PyTree = Any


@filter_jit
def _minimize(
    fit_objective: Callable[[Array], float],
    model: PyTree,
    data: Array,
    init_pars: Array,
    bounds: Array,
    method: str = "LBFGSB",
    maxiter: int = 500,
    tol: float = 1e-6,
    other_settings: dict[str, float] | None = None,
):
    other_settings = other_settings or {}
    other_settings["maxiter"] = maxiter
    other_settings["tol"] = tol
    minimizer = getattr(jaxopt, method)(
        fun=fit_objective, implicit_diff=True, **other_settings
    )
    if "bounds" in inspect.signature(minimizer.init_state).parameters:
        return minimizer.run(init_pars, bounds=bounds, model=model, data=data)[0]
    return minimizer.run(init_pars, model=model, data=data)[0]


@filter_jit
def fit(
    data: Array,
    model: PyTree,
    init_pars: Array | None = None,
    bounds: tuple[Array, Array] | None = None,
    method: str = "LBFGSB",
    maxiter: int = 500,
    tol: float = 1e-6,
    other_settings: dict[str, float] | None = None,
) -> Array:
    def fit_objective(pars: Array, model: PyTree, data: Array) -> float:
        return cast(float, -model.logpdf(pars, data)[0])

    if bounds is None:
        bounds = model.config.suggested_bounds()

    if init_pars is None:
        init_pars = model.config.suggested_init()

    return _minimize(
        fit_objective=fit_objective,
        model=model,
        data=data,
        init_pars=init_pars,
        bounds=bounds,
        method=method,
        maxiter=maxiter,
        tol=tol,
        other_settings=other_settings,
    )


@filter_jit
def fixed_poi_fit(
    data: Array,
    model: PyTree,
    poi_condition: float,
    init_pars: Array | None = None,
    bounds: Array | None = None,
    method: str = "LBFGSB",
    maxiter: int = 500,
    tol: float = 1e-6,
    other_settings: dict[str, float] | None = None,
) -> Array:
    poi_idx = model.config.poi_index

    def fit_objective(pars: Array, model: PyTree, data: Array) -> float:  # NLL
        """lhood_pars_to_optimize: either all pars, or just nuisance pars"""
        # pyhf.Model.logpdf returns list[float]
        blank = jnp.zeros_like(jnp.asarray(model.config.suggested_init()))
        blank += pars
        return cast(float, -model.logpdf(blank.at[poi_idx].set(poi_condition), data)[0])

    if bounds is None:
        lower, upper = model.config.suggested_bounds()
        # ignore poi bounds
        upper = jnp.delete(upper, poi_idx)
        lower = jnp.delete(lower, poi_idx)
        bounds = jnp.array([lower, upper])

    if init_pars is None:
        init_pars = model.config.suggested_init()
        # ignore poi init
        init_pars = jnp.delete(init_pars, poi_idx)

    fit_res = _minimize(
        fit_objective=fit_objective,
        model=model,
        data=data,
        init_pars=init_pars,
        bounds=bounds,
        method=method,
        maxiter=maxiter,
        tol=tol,
        other_settings=other_settings,
    )
    blank = jnp.zeros_like(jnp.asarray(model.config.suggested_init()))
    blank += fit_res
    poi_idx = model.config.poi_index
    return blank.at[poi_idx].set(poi_condition)
