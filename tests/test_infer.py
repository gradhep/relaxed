from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pyhf
import pytest
from dummy_pyhf import example_model
from jax import jacrev

import relaxed

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("test_stat", ["q", "q0"])
@pytest.mark.parametrize("phi", np.linspace(0.0, 10.0, 5))
def test_hypotest_validity(phi, test_stat):
    pyhf.set_backend("jax")
    if test_stat == "q":
        analytic_pars = jnp.array([0.0, 1.0])  # bkg-only hypothesis
        analytic_pars_dict = {"mu": 0.0, "shapesys": 1.0}
    elif test_stat == "q0":
        analytic_pars = jnp.array([1.0, 1.0])  # nominal sig+bkg hypothesis
        analytic_pars_dict = {"mu": 1.0, "shapesys": 1.0}
    else:
        msg = f"Unknown test statistic: {test_stat}"
        raise ValueError(msg)
    model, yields = example_model(phi, return_yields=True, n_bins=1)
    relaxed_cls = relaxed.infer.hypotest(
        test_poi=1,
        poi_name="mu",
        data=model.expected_data(analytic_pars_dict),
        model=model,
        init_pars={"mu": 1.0, "shapesys": 1.0},
        test_stat=test_stat,
    )
    m = pyhf.simplemodels.uncorrelated_background(*yields)
    pyhf_cls = pyhf.infer.hypotest(
        1, m.expected_data(analytic_pars), m, test_stat=test_stat
    )
    assert np.allclose(
        relaxed_cls,
        pyhf_cls,
    )


@pytest.mark.parametrize("test_stat", ["q", "q0"])
def test_hypotest_expected(test_stat):
    pyhf.set_backend("jax")
    if test_stat == "q":
        analytic_pars = jnp.array([0.0, 1.0])  # bkg-only hypothesis
        analytic_pars_dict = {"mu": 0.0, "shapesys": 1.0}
    elif test_stat == "q0":
        analytic_pars = jnp.array([1.0, 1.0])  # nominal sig+bkg hypothesis
        analytic_pars_dict = {"mu": 1.0, "shapesys": 1.0}
    else:
        msg = f"Unknown test statistic: {test_stat}"
        raise ValueError(msg)
    model, yields = example_model(5.0, return_yields=True, n_bins=1)
    relaxed_cls = relaxed.infer.hypotest(
        test_poi=1,
        poi_name="mu",
        data=model.expected_data(analytic_pars_dict),
        model=model,
        init_pars={"mu": 1.0, "shapesys": 1.0},
        test_stat=test_stat,
        expected_pars=analytic_pars_dict,
    )
    m = pyhf.simplemodels.uncorrelated_background(*yields)
    pyhf_cls = pyhf.infer.hypotest(
        1, m.expected_data(analytic_pars), m, test_stat=test_stat
    )
    assert np.allclose(
        relaxed_cls,
        pyhf_cls,
    )


@pytest.mark.parametrize("test_stat", ["q", "q0"])
@pytest.mark.parametrize("expected_pars", [True, False])
def test_hypotest_grad(test_stat, expected_pars):
    pars = {"mu": 0.0, "shapesys": 1.0}
    expars = pars if expected_pars else None

    def pipeline(x):
        model = example_model(x * 5.0, n_bins=1)
        return relaxed.infer.hypotest(
            1.0,
            init_pars={"mu": 1.0, "shapesys": 1.0},
            model=model,
            data=model.expected_data(pars),
            test_stat=test_stat,
            expected_pars=expars,
        )

    jacrev(pipeline)(jnp.asarray(0.5))


@pytest.mark.parametrize("expected_pars", [True, False])
def test_hypotest_grad_noCLs(expected_pars):
    pars = {"mu": 0.0, "shapesys": 1.0}
    expars = pars if expected_pars else None

    def pipeline(x):
        model = example_model(x * 5.0, n_bins=1)
        return relaxed.infer.hypotest(
            1.0,
            init_pars={"mu": 1.0, "shapesys": 1.0},
            model=model,
            data=model.expected_data(pars),
            test_stat="qmu",
            expected_pars=expars,
            cls_method=False,
        )

    jacrev(pipeline)(jnp.asarray(0.5))


def test_wrong_test_stat():
    model = example_model(0.0, n_bins=1)
    with pytest.raises(ValueError, match="Unknown test statistic: q1"):
        relaxed.infer.hypotest(
            1,
            data=model.expected_data({"mu": 0.0, "shapesys": 1.0}),
            model=model,
            test_stat="q1",
            init_pars={"mu": 1.0, "shapesys": 1.0},
        )
