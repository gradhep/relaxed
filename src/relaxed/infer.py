"""Calculate expected CLs values with hypothesis tests."""
from __future__ import annotations

__all__ = ("hypotest",)

import logging
from typing import Any, TYPE_CHECKING

import jax.numpy as jnp
from equinox import filter_jit
from jax import Array
import jax.scipy as jsp

from relaxed.mle import fit, fixed_poi_fit

if TYPE_CHECKING:
    PyTree = Any
    from jax.typing import ArrayLike


@filter_jit
def hypotest(
    test_poi: float,
    data: Array,
    model: PyTree,
    init_pars: dict[str, ArrayLike],
    bounds: dict[str, ArrayLike],
    poi_name: str = "mu",
    return_mle_pars: bool = False,
    test_stat: str = "qmu",
    expected_pars: dict[str, ArrayLike] | None = None,
    cls_method: bool = True,
) -> tuple[Array, Array] | Array:
    """Calculate expected CLs/p-values via hypothesis tests.

    Parameters
    ----------
    test_poi : float
        The value of the test parameter to use for the hypothesis test.
    data : Array
        The data to use for the hypothesis test.
    model : PyTree
        The model to use for the hypothesis test.
    init_pars : dict[str, ArrayLike]
        The initial parameters to use for fits within the hypothesis test.
    bounds : dict[str, ArrayLike]
        The bounds to use for fits within the hypothesis test.
    poi_name : str
        The name of the parameter(s) of interest.
    return_mle_pars : bool
        Whether to return the MLE parameters.
    test_stat : str
        The test statistic type to use for the hypothesis test. Default is `qmu`.
    expected_pars : dict[str, ArrayLike] | None
        The MLE parameters from a previous fit, to use as the expected parameters.
    cls_method : bool
        Whether to use the CLs method for the hypothesis test. Default is True (if qmu test)

    Returns
    -------
    Array
        The expected CLs/p-value.
    or tuple[Array, Array]
        The expected CLs/p-value and the MLE parameters. Only returned if `return_mle_pars` is True.
    """
    if test_stat == "q" or test_stat == "qmu":
        return qmu_test(
            test_poi, data, model, init_pars, bounds, poi_name, return_mle_pars, expected_pars, cls_method
        )
    if test_stat == "q0":
        logging.info(
            "test_poi automatically set to 0 for q0 test (bkg-only null hypothesis)"
        )
        return q0_test(0.0, data, model, init_pars, bounds, poi_name, return_mle_pars, expected_pars)

    msg = f"Unknown test statistic: {test_stat}"
    raise ValueError(msg)


@filter_jit
def qmu_test(
    test_poi: float,
    data: Array,
    model: PyTree,
    init_pars: dict[str, ArrayLike],
    bounds: dict[str, ArrayLike],
    poi_name: str,
    return_mle_pars: bool = False,
    expected_pars: Array | None = None,
    cls_method: bool = True,
) -> tuple[Array, Array] | Array:
    # remove the poi from the init_pars
    conditional_init = {k: v for k, v in init_pars.items() if k != poi_name}
    conditional_bounds = {k: v for k, v in bounds.items() if k != poi_name}
    conditional_pars = fixed_poi_fit(
        data, model, poi_value=test_poi, poi_name=poi_name, init_pars=conditional_init, bounds=conditional_bounds
    )
    if expected_pars is None:
        mle_pars = fit(data, model, init_pars=init_pars, bounds=bounds)
    else:
        mle_pars = expected_pars
    profile_likelihood = -2 * (
        model.logpdf(pars=conditional_pars, data=data) - model.logpdf(pars=mle_pars, data=data)
    )

    poi_hat = mle_pars[poi_name]
    qmu = jnp.where(poi_hat < test_poi, profile_likelihood, 0.0)

    CLsb = 1 - jsp.stats.norm.cdf(jnp.sqrt(qmu), loc=0, scale=1)
    if cls_method:
        altval = 0.0
        CLb = 1 - jsp.stats.norm.cdf(altval, loc=0, scale=1)
        CLs = CLsb / CLb
    else:
        CLs = CLsb
    return (CLs, mle_pars) if return_mle_pars else CLs


@filter_jit
def q0_test(
    test_poi: float,
    data: Array,
    model: PyTree,
    init_pars: dict[str, ArrayLike],
    bounds: dict[str, ArrayLike],
    poi_name: str,
    return_mle_pars: bool = False,
    expected_pars: Array | None = None,
) -> tuple[Array, Array] | Array:
    # remove the poi from the init_pars
    conditional_init = {k: v for k, v in init_pars.items() if k != poi_name}
    conditional_bounds = {k: v for k, v in bounds.items() if k != poi_name}
    conditional_pars = fixed_poi_fit(
        data, model, poi_value=test_poi, poi_name=poi_name, init_pars=conditional_init, bounds=conditional_bounds
    )
    if expected_pars is None:
        mle_pars = fit(data, model, init_pars=init_pars, bounds=bounds)
    else:
        mle_pars = expected_pars
    profile_likelihood = -2 * (
        model.logpdf(pars=conditional_pars, data=data) - model.logpdf(pars=mle_pars, data=data)
    )

    poi_hat = mle_pars[poi_name]
    q0 = jnp.where(poi_hat >= test_poi, profile_likelihood, 0.0)
    p0 = 1 - jsp.stats.norm.cdf(jnp.sqrt(q0))

    return (p0, mle_pars) if return_mle_pars else p0
