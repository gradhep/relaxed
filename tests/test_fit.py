import jax.numpy as jnp
import numpy as np
from dummy_pyhf import uncorrelated_background
from jax import jacrev

import relaxed.mle


def test_fit(example_model):
    analytic_pars = jnp.array([0.0, 1.0])

    mle_pars = relaxed.mle.fit(
        model=example_model,
        data=example_model.expected_data(analytic_pars),
        init_pars=example_model.config.suggested_init(),
        lr=1e-2,
    )
    assert np.allclose(mle_pars, analytic_pars, atol=0.01)


def test_fit_grad():
    def pipeline(x):
        model = uncorrelated_background(x * 5.0, x * 20, x * 2)
        mle_pars = relaxed.mle.fit(
            model=model,
            data=model.expected_data(jnp.array([0.0, 1.0])),
            init_pars=model.config.suggested_init(),
            lr=1e-2,
        )
        return mle_pars

    jacrev(pipeline)(jnp.asarray(0.5))
