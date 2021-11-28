import jax.numpy as jnp
import numpy as np
import pyhf
from dummy_pyhf import uncorrelated_background
from jax import jacrev

import relaxed


def test_hypotest_validity(example_model):
    pyhf.set_backend("jax")
    analytic_pars = jnp.array([0.0, 1.0])
    relaxed_cls = relaxed.infer.hypotest(
        1, example_model.expected_data(analytic_pars), example_model, lr=1e-2
    )
    m = pyhf.simplemodels.uncorrelated_background([5], [50], [5])
    pyhf_cls = pyhf.infer.hypotest(1, m.expected_data(analytic_pars), m)
    np.allclose(relaxed_cls, pyhf_cls)


def test_hypotest_grad():
    def pipeline(x):
        model = uncorrelated_background(x * 5.0, x * 20, x * 2)
        expected_cls = relaxed.infer.hypotest(
            1.0,
            model=model,
            data=model.expected_data(jnp.array([0.0, 1.0])),
            lr=1e-2,
        )
        return expected_cls

    jacrev(pipeline)(jnp.asarray(0.5))
