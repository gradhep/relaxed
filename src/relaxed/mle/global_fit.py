from __future__ import annotations

__all__ = ("fit",)

from typing import TYPE_CHECKING, Callable, cast

from chex import Array

if TYPE_CHECKING:
    import pyhf

from .minimize import _minimize


def global_fit_objective(data: Array, model: pyhf.Model) -> Callable[[Array], float]:
    def fit_objective(lhood_pars_to_optimize: Array) -> float:  # NLL
        """lhood_pars_to_optimize: either all pars, or just nuisance pars"""
        return cast(
            float, -model.logpdf(lhood_pars_to_optimize, data)[0]
        )  # pyhf.Model.logpdf returns list[float]

    return fit_objective


# @partial(jax.jit, static_argnames=["model"]) # forward pass
def fit(
    data: Array,
    model: pyhf.Model,
    init_pars: Array,
    lr: float = 4e-3,
) -> Array:
    obj = global_fit_objective(data, model)
    fit_res = _minimize(obj, init_pars, lr)
    return fit_res
