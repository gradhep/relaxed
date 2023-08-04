from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pyhf
import pytest
from dummy_pyhf import example_model
from jax import config, jacrev, tree_util

import relaxed

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("phi", np.linspace(0.0, 10.0, 5))
def test_fit(phi):
    analytic_pars = {"mu": 0.0, "shapesys": 1.0}
    model = example_model(phi, n_bins=1)
    mle_pars = relaxed.mle.fit(
        model=model,
        data=model.expected_data(analytic_pars),
        init_pars={"mu": 1.0, "shapesys": 1.0},
        bounds={"mu": (-1, 10), "shapesys": (-1, 10)},
    )
    assert np.allclose(
        tree_util.tree_flatten(mle_pars)[0], tree_util.tree_flatten(analytic_pars)[0]
    )


def test_fit_grad():
    def pipeline(x):
        model = example_model(x, n_bins=2)
        pars = {"mu": jnp.array(0.0), "shapesys": jnp.array([1.0, 1.0])}

        return relaxed.infer.hypotest(
            1.0,
            data=model.expected_data(pars),
            model=model,
            init_pars={"mu": jnp.array(1.0), "shapesys": jnp.array([1.0, 1.0])},
            bounds={"mu": (0, 10), "shapesys": (0, 10)},
        )

    jacrev(pipeline)(jnp.asarray(0.5))


@pytest.mark.parametrize("phi", np.linspace(0.0, 10.0, 5))
def test_fixed_poi_fit(phi):
    pars = {"mu": 0.0, "shapesys": 1.0}
    model, yields = example_model(phi, return_yields=True, n_bins=1)
    init = {"shapesys": 1.0}

    relaxed_mle = relaxed.mle.fixed_poi_fit(
        model=model,
        data=model.expected_data(pars),
        init_pars=init,
        poi_value=1.0,
        poi_name="mu",
    )
    pyhf.set_backend("jax")
    m = pyhf.simplemodels.uncorrelated_background(*yields)
    analytic_pars = jnp.array(m.config.suggested_init()).at[m.config.poi_index].set(0.0)

    pyhf_mle = pyhf.infer.mle.fixed_poi_fit(
        1.0,
        m.expected_data(analytic_pars),
        m,
    )

    assert np.allclose(tree_util.tree_flatten(relaxed_mle)[0], pyhf_mle, atol=1e-4)


def test_fixed_poi_fit_grad():
    def pipeline(x):
        model = example_model(x, n_bins=2)
        pars = {"mu": jnp.array(0.0), "shapesys": jnp.array([1.0, 1.0])}

        return relaxed.infer.hypotest(
            1.0,
            data=model.expected_data(pars),
            model=model,
            init_pars={"mu": jnp.array(1.0), "shapesys": jnp.array([1.0, 1.0])},
            bounds={"mu": (0, 10), "shapesys": (0, 10)},
        )

    jacrev(pipeline)(jnp.asarray(0.5))
