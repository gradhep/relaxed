from __future__ import annotations

from typing import Any, Callable

__all__ = ("_minimize",)

from functools import partial

import jax
import jaxopt
import optax
from chex import Array


# try wrapping obj with closure_convert
@partial(jax.jit, static_argnames=["objective_fn"])  # forward pass
def _minimize(
    objective_fn: Callable[..., float], init_pars: Array, lr: float, *obj_args: Any
) -> Array:
    converted_fn, aux_pars = jax.closure_convert(objective_fn, init_pars, *obj_args)
    # aux_pars seems to be empty? took that line from jax docs example...
    solver = jaxopt.OptaxSolver(
        fun=converted_fn, opt=optax.adam(lr), implicit_diff=True, maxiter=5000
    )
    return solver.run(init_pars, *obj_args, *aux_pars)[0]
