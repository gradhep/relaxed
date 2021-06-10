"""instantiate routines for fitting a model given a fixed param of intrest."""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.experimental.optimizers as optimizers
import jax.numpy as jnp
from fax.implicit import two_phase_solver

from .._types import ArrayDevice
from .minuit_transforms import to_bounded_vec, to_inf


def constrained_fit(
    model_maker: Callable[..., tuple[Any, ArrayDevice]],
    model_kwargs: dict[str, Any] = dict(),
    pdf_transform: bool = False,
    default_rtol: float = 1e-10,
    default_atol: float = 1e-10,
    default_max_iter: int = int(1e7),
    learning_rate: float = 1e-6,
) -> Callable[[ArrayDevice, tuple[ArrayDevice, float]], ArrayDevice]:
    """Initialize a constrained fit of a model via gradient descent w/ adam optimizer.

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
    constrained_fitter: function that peforms the fit in a differentiable way
    via Christianson's two-phase method.
    Takes in (initial params, model hyperparams) as args.
    """
    adam_init, adam_update, adam_get_params = optimizers.adam(learning_rate)

    def make_model(
        hyper_pars: tuple[ArrayDevice, float],
    ) -> tuple[float, Callable[[ArrayDevice], float]]:

        model_pars, constrained_mu = hyper_pars
        m, bonlypars = model_maker(model_pars, **model_kwargs)

        bounds = jnp.array(m.config.suggested_bounds())
        constrained_mu = (
            to_inf(constrained_mu, bounds[0]) if pdf_transform else constrained_mu
        )

        exp_bonly_data = m.expected_data(bonlypars, include_auxdata=True)

        def expected_logpdf(
            pars: ArrayDevice,
        ) -> tuple[float]:  # maps pars to bounded space if pdf_transform = True

            return (
                m.logpdf(to_bounded_vec(pars, bounds), exp_bonly_data)
                if pdf_transform
                else m.logpdf(pars, exp_bonly_data)
            )

        def constrained_fit_objective(nuis_par: ArrayDevice) -> float:  # NLL
            pars = jnp.concatenate([jnp.asarray([constrained_mu]), nuis_par])
            return -expected_logpdf(pars)[0]

        return constrained_mu, constrained_fit_objective

    def constrained_bestfit_minimized(
        hyper_pars: tuple[ArrayDevice, float],
    ) -> Callable[[int, ArrayDevice], ArrayDevice]:
        mu, cnll = make_model(hyper_pars)

        def bestfit_via_grad_descent(
            i: int, param: ArrayDevice
        ) -> ArrayDevice:  # gradient descent
            _, np = param[0], param[1:]
            grads = jax.grad(cnll)(np)
            np = adam_get_params(adam_update(i, grads, adam_init(np)))
            param = jnp.concatenate([jnp.asarray([mu]), np])
            return param

        return bestfit_via_grad_descent

    constrained_solver = two_phase_solver(
        param_func=constrained_bestfit_minimized,
        default_rtol=default_rtol,
        default_atol=default_atol,
        default_max_iter=default_max_iter,
    )

    def constrained_fitter(
        init: ArrayDevice, hyper_pars: tuple[ArrayDevice, float]
    ) -> ArrayDevice:
        solve = constrained_solver(init, hyper_pars)
        return solve.value

    return constrained_fitter
