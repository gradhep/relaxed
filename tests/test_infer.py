import jax.numpy as jnp
import numpy as np
import pyhf
import pytest
from dummy_pyhf import example_model, uncorrelated_background
from jax import jacrev

import relaxed


@pytest.mark.parametrize("test_stat", ["q", "q0"])
@pytest.mark.parametrize("phi", np.linspace(0.0, 10.0, 5))
def test_hypotest_validity(phi, test_stat):
    pyhf.set_backend("jax")
    if test_stat == "q":
        analytic_pars = jnp.array([0.0, 1.0])  # bkg-only hypothesis
    elif test_stat == "q0":
        analytic_pars = jnp.array([1.0, 1.0])  # nominal sig+bkg hypothesis
    else:
        raise ValueError(f"Unknown test statistic: {test_stat}")
    model, yields = example_model(phi, return_yields=True)
    relaxed_cls = relaxed.infer.hypotest(
        1, model.expected_data(analytic_pars), model, lr=1e-3, test_stat=test_stat
    )
    m = pyhf.simplemodels.uncorrelated_background(*yields)
    pyhf_cls = pyhf.infer.hypotest(
        1, m.expected_data(analytic_pars), m, test_stat=test_stat
    )
    assert np.allclose(
        relaxed_cls,
        pyhf_cls,
        atol=0.001,
    )  # tested working without dummy_pyhf on a pyhf fork, but not main yet


@pytest.mark.parametrize("test_stat", ["q", "q0"])
def test_hypotest_expected(test_stat):
    pyhf.set_backend("jax")
    if test_stat == "q":
        analytic_pars = jnp.array([0.0, 1.0])  # bkg-only hypothesis
    elif test_stat == "q0":
        analytic_pars = jnp.array([1.0, 1.0])  # nominal sig+bkg hypothesis
    else:
        raise ValueError(f"Unknown test statistic: {test_stat}")
    model, yields = example_model(5.0, return_yields=True)
    relaxed_cls = relaxed.infer.hypotest(
        1,
        model.expected_data(analytic_pars),
        model,
        lr=1e-3,
        test_stat=test_stat,
        expected_pars=analytic_pars,
    )
    m = pyhf.simplemodels.uncorrelated_background(*yields)
    pyhf_cls = pyhf.infer.hypotest(
        1, m.expected_data(analytic_pars), m, test_stat=test_stat
    )
    assert np.allclose(
        relaxed_cls,
        pyhf_cls,
        atol=0.001,
    )  # tested working without dummy_pyhf on a pyhf fork, but not main yet


@pytest.mark.parametrize("test_stat", ["q", "q0"])
@pytest.mark.parametrize("expected_pars", [True, False])
def test_hypotest_grad(test_stat, expected_pars):
    pars = jnp.array([0.0, 1.0])
    if expected_pars:
        expars = pars
    else:
        expars = None

    def pipeline(x):
        model = uncorrelated_background(x * 5.0, x * 20, x * 2)
        expected_cls = relaxed.infer.hypotest(
            1.0,
            model=model,
            data=model.expected_data(pars),
            lr=1e-2,
            test_stat=test_stat,
            expected_pars=expars,
        )
        return expected_cls

    jacrev(pipeline)(jnp.asarray(0.5))


def test_wrong_test_stat():
    with pytest.raises(ValueError):
        model = example_model(0.0)
        relaxed.infer.hypotest(
            1,
            model.expected_data(jnp.array([0.0, 1.0])),
            model,
            lr=1e-2,
            test_stat="q1",
        )
