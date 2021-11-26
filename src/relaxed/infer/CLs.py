"""Calculate expected CLs values with hypothesis tests."""
from __future__ import annotations

__all__ = ("hypotest",)


import jax.numpy as jnp
import pyhf
from chex import Array

from ..mle import fit, fixed_poi_fit


# @partial(jax.jit, static_argnames=["model", "return_mle_pars"]) # forward pass
def hypotest(
    test_poi: float,
    data: Array,
    model: pyhf.Model,
    init_pars: Array,
    return_mle_pars: bool = False,
) -> Array | tuple[Array, Array]:

    conditional_pars = fixed_poi_fit(
        data,
        model,
        poi_condition=test_poi,
        init_pars=init_pars,
    )
    mle_pars = fit(data, model, init_pars=init_pars)

    profile_likelihood = -2 * (
        model.logpdf(conditional_pars, data)[0] - model.logpdf(mle_pars, data)[0]
    )

    poi_hat = mle_pars[model.config.poi_index]
    qmu = jnp.where(poi_hat < test_poi, profile_likelihood, 0.0)

    CLsb = 1 - pyhf.tensorlib.normal_cdf(jnp.sqrt(qmu))
    altval = 0.0
    CLb = 1 - pyhf.tensorlib.normal_cdf(altval)
    CLs = CLsb / CLb
    return CLs, mle_pars if return_mle_pars else CLs