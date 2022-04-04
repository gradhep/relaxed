import jax.numpy as jnp
import pyhf
from dummy_pyhf import example_model
from jax import jacrev
from jax.random import PRNGKey

import relaxed


def test_gaussianity():
    pyhf.set_backend("jax")
    m = pyhf.simplemodels.uncorrelated_background([5, 5], [50, 50], [5, 5])
    pars = jnp.asarray(m.config.suggested_init())
    data = jnp.asarray(m.expected_data(pars))
    relaxed.gaussianity(m, pars, data, PRNGKey(0))


def test_gaussianity_grad():
    def pipeline(x):
        model = example_model(5.0)
        pars = model.config.suggested_init()
        data = model.expected_data(pars)
        return relaxed.metrics.gaussianity(model, pars * x, data * x, PRNGKey(0))

    jacrev(pipeline)(4.0)  # just check you can calc it w/o exception
