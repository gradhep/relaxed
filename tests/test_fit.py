import jax.numpy as jnp
import numpy as np
import pyhf
import pytest
from dummy_pyhf import example_model, uncorrelated_background
from jax import jacrev

import relaxed


@pytest.mark.parametrize("phi", np.linspace(0.0, 10.0, 5))
def test_fit(phi):
    analytic_pars = jnp.array([0.0, 1.0])
    model = example_model(phi)
    mle_pars = relaxed.mle.fit(
        model=model,
        data=model.expected_data(analytic_pars),
        init_pars=model.config.suggested_init(),
        lr=1e-3,
    )
    assert np.allclose(mle_pars, analytic_pars, atol=0.05)


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


@pytest.mark.parametrize("phi", np.linspace(0.0, 10.0, 5))
def test_fixed_poi_fit(phi):
    pyhf.set_backend("jax")
    analytic_pars = jnp.array([0.0, 1.0])

    model, yields = example_model(phi, return_yields=True)
    init = np.asarray(model.config.suggested_init())
    init = jnp.asarray(np.delete(init, model.config.poi_index))
    relaxed_mle = relaxed.mle.fixed_poi_fit(
        model=model,
        data=model.expected_data(analytic_pars),
        init_pars=init,
        lr=1e-2,
        poi_condition=1.0,
    )

    m = pyhf.simplemodels.uncorrelated_background(*yields)

    pyhf_mle = pyhf.infer.mle.fixed_poi_fit(
        1.0,
        m.expected_data(analytic_pars),
        m,
    )

    assert np.allclose(relaxed_mle, pyhf_mle, atol=0.05)


def test_fixed_poi_fit_grad():
    def pipeline(x):
        model = uncorrelated_background(x * 5.0, x * 20, x * 2)
        mle_pars = relaxed.mle.fixed_poi_fit(
            model=model,
            data=model.expected_data(jnp.array([0.0, 1.0])),
            init_pars=model.config.suggested_init()[1:],
            lr=1e-2,
            poi_condition=1.0,
        )
        return mle_pars

    jacrev(pipeline)(jnp.asarray(0.5))
