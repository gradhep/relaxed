"""instantiate routines for fitting a model given a fixed param of intrest."""
from __future__ import annotations

from typing import Any, Callable

__all__ = ["global_fit"]

import jax
import jax.experimental.optimizers as optimizers
import jax.numpy as jnp
from fax.implicit import two_phase_solver

from .._types import ArrayDevice
from .minuit_transforms import to_bounded_vec, to_inf


def global_fit(
    model_maker: Callable[..., tuple[Any, ArrayDevice]],
    model_kwargs: dict[str, Any] = dict(),
    pdf_transform: bool = False,
    default_rtol: float = 1e-10,
    default_atol: float = 1e-10,
    default_max_iter: int = int(1e7),
    learning_rate: float = 1e-6,
) -> Callable[[ArrayDevice, tuple[ArrayDevice, float]], ArrayDevice]:
    """Initialize a fit of a model via gradient descent w/ adam optimizer.
    Parameters
    ----------
    model_maker: function that takes model params and returns the tuple
    [model, background-only parameters] when called.
    pdf_transform: whether to map the likelihood domain to [-inf, inf]
    during minimization.
    default_rtol, default_atol, default_max_iter: args to pass to fax's
    two-phase solver.
    learning_rate: controls step-size in grad descent. arg supplied to adam optimizer.
    Returns
    -------
    global_fitter: function that peforms the fit in a differentiable way
    via Christianson's two-phase method.
    Takes in (initial params, model hyperparams) as args.
    """
    adam_init, adam_update, adam_get_params = optimizers.adam(learning_rate)

    def make_model(
        hyper_pars: tuple[ArrayDevice, float],
    ) -> tuple[float, Callable[[ArrayDevice], float]]:

        m, bonlypars = model_maker(hyper_pars, **model_kwargs)

        bounds = jnp.array(m.config.suggested_bounds())

        exp_bonly_data = m.expected_data(bonlypars, include_auxdata=True)

        def expected_logpdf(
            pars: ArrayDevice,
        ) -> tuple[float]:  # maps pars to bounded space if pdf_transform = True

            return (
                m.logpdf(to_bounded_vec(pars, bounds), exp_bonly_data)
                if pdf_transform
                else m.logpdf(pars, exp_bonly_data)
            )

        def global_fit_objective(pars: ArrayDevice) -> float:  # NLL
            return -expected_logpdf(pars)[0]

        return global_fit_objective

    def global_bestfit_minimized(
        hyper_pars: tuple[ArrayDevice, float],
    ) -> Callable[[int, ArrayDevice], ArrayDevice]:
        cnll = make_model(hyper_pars)

        def bestfit_via_grad_descent(
            i: int, params: ArrayDevice
        ) -> ArrayDevice:  # gradient descent
            grads = jax.grad(cnll)(params)
            np = adam_get_params(adam_update(i, grads, adam_init(params)))
            return params

        return bestfit_via_grad_descent

    global_solver = two_phase_solver(
        param_func=global_bestfit_minimized,
        default_rtol=default_rtol,
        default_atol=default_atol,
        default_max_iter=default_max_iter,
    )

    def global_fitter(
        init: ArrayDevice, hyper_pars: tuple[ArrayDevice, float]
    ) -> ArrayDevice:
        solve = global_solver(init, hyper_pars)
        return solve.value

    return global_fitter
