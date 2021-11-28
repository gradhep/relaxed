import jax.numpy as jnp
import numpy as np
import pyhf
from dummy_pyhf import uncorrelated_background
from jax import jacrev

import relaxed


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


def test_fixed_poi_fit():
    pyhf.set_backend("jax")
    analytic_pars = jnp.array([0.0, 1.0])

    example_model = uncorrelated_background(
        signal_data=jnp.asarray([5]),
        bkg_data=jnp.asarray([50]),
        bkg_uncerts=jnp.asarray([5]),
    )
    init = np.asarray(example_model.config.suggested_init())
    init = jnp.asarray(np.delete(init, example_model.config.poi_index))
    relaxed_mle = relaxed.mle.fixed_poi_fit(
        model=example_model,
        data=example_model.expected_data(analytic_pars),
        init_pars=init,
        lr=1e-2,
        poi_condition=1.0,
    )

    m = pyhf.simplemodels.uncorrelated_background([5], [50], [5])

    pyhf_mle = pyhf.infer.mle.fixed_poi_fit(
        1.0,
        m.expected_data(analytic_pars),
        m,
    )

    assert np.allclose(relaxed_mle, pyhf_mle, rtol=1e-4)


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
