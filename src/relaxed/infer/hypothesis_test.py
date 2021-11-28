"""Calculate expected CLs values with hypothesis tests."""
from __future__ import annotations

__all__ = ("hypotest",)

from functools import partial

import jax
import jax.numpy as jnp
import pyhf
from chex import Array

from ..mle import fit, fixed_poi_fit


@partial(jax.jit, static_argnames=["model", "return_mle_pars"])  # forward pass
def hypotest(
    test_poi: float,
    data: Array,
    model: pyhf.Model,
    lr: float,
    return_mle_pars: bool = False,
) -> tuple[Array, Array] | Array:
    # hard-code 1 as inits for now
    # TODO: need to parse different inits for constrained and global fits
    init_pars = jnp.ones_like(jnp.asarray(model.config.suggested_init()))
    conditional_pars = fixed_poi_fit(
        data, model, poi_condition=test_poi, init_pars=init_pars[:-1], lr=lr
    )
    mle_pars = fit(data, model, init_pars=init_pars, lr=lr)
    profile_likelihood = -2 * (
        model.logpdf(conditional_pars, data)[0] - model.logpdf(mle_pars, data)[0]
    )

    poi_hat = mle_pars[model.config.poi_index]
    qmu = jnp.where(poi_hat < test_poi, profile_likelihood, 0.0)

    CLsb = 1 - pyhf.tensorlib.normal_cdf(jnp.sqrt(qmu))
    altval = 0.0
    CLb = 1 - pyhf.tensorlib.normal_cdf(altval)
    CLs = CLsb / CLb
    return (CLs, mle_pars) if return_mle_pars else CLs
