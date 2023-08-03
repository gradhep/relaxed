from __future__ import annotations

__all__ = ("fit", "fixed_poi_fit")

import inspect
from typing import TYPE_CHECKING, Any, Callable, cast

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from equinox import filter_jit
from jax import Array

if TYPE_CHECKING:
    from jax.typing import ArrayLike

    PyTree = Any


def _parse_bounds(
    bounds: dict[str, ArrayLike], init_pars: dict[str, ArrayLike]
) -> tuple[dict[str, ArrayLike], dict[str, ArrayLike]]:
    """Convert dict of bounds to a dict of lower and a dict of upper bounds."""
    lower = {}
    upper = {}

    for k, v in bounds.items():
        # Convert to array for easy manipulation
        array_v = jnp.asarray(v)

        # Check if v is 1D or 2D
        if array_v.ndim == 1:
            if (
                isinstance(init_pars[k], (list, jax.Array, np.ndarray))
                and init_pars[k].size > 1
            ):  # If the initial parameter is a list or array
                lower[k] = jnp.array([array_v[0]] * len(init_pars[k]))
                upper[k] = jnp.array([array_v[1]] * len(init_pars[k]))
            else:  # If the initial parameter is a single value
                lower[k] = array_v[0]
                upper[k] = array_v[1]
        else:
            lower[k] = jnp.array([item[0] for item in array_v])
            upper[k] = jnp.array([item[1] for item in array_v])

    return lower, upper


@filter_jit
def _minimize(
    fit_objective: Callable[[Array], float],
    model: PyTree,
    data: Array,
    init_pars: dict[str, ArrayLike],
    bounds: dict[str, ArrayLike],
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
        if bounds is not None:
            bounds = _parse_bounds(bounds, init_pars)
        return minimizer.run(init_pars, bounds=bounds, model=model, data=data)[0]
    return minimizer.run(init_pars, model=model, data=data)[0]


@filter_jit
def fit(
    data: Array,
    model: PyTree,
    init_pars: dict[str, ArrayLike],
    bounds: dict[str, Array] | None = None,
    method: str = "LBFGSB",
    maxiter: int = 500,
    tol: float = 1e-6,
    other_settings: dict[str, float] | None = None,
) -> dict[str, Array]:
    def fit_objective(pars: Array, model: PyTree, data: Array) -> float:
        return cast(float, -model.logpdf(data=data, pars=pars))

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
    poi_value: float,
    poi_name: str,
    init_pars: dict[str, ArrayLike],
    bounds: dict[str, Array] | None = None,
    method: str = "LBFGSB",
    maxiter: int = 500,
    tol: float = 1e-6,
    other_settings: dict[str, float] | None = None,
) -> dict[str, Array]:
    def fit_objective(
        pars: dict[str, Array], model: PyTree, data: Array
    ) -> float:  # NLL
        """lhood_pars_to_optimize: either all pars, or just nuisance pars"""
        pars[poi_name] = poi_value
        return cast(float, -model.logpdf(data=data, pars=pars))

    res = _minimize(
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
    res[poi_name] = poi_value
    return res
