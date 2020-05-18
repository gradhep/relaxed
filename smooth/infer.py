__all__ = ["expected_pvalue_upper_limit", "CLs"]

import jax
import jax.numpy as jnp
from jax import config
from jax.lax import custom_root
import pyhf
from functools import partial

# avoid those precision errors!
config.update("jax_enable_x64", True)

pyhf.set_backend(pyhf.tensor.jax_backend())

from .fit import global_fit, constrained_fit
from .transforms import *


def expected_pvalue_upper_limit(model_maker, solver_kwargs):
    """
    Args:
            model_maker: Function that returns a Model object using the `params` arg.

    Returns:
            cls_jax: A callable function that takes the parameters of the observable as argument,
            and returns an expected p-value from testing the background-only model against the
            nominal signal hypothesis (or whatever corresponds to the value of the arg 'test_mu')
    """

    @jax.jit
    def pmu_jax(test_mu, params):
        g_fitter = global_fit(model_maker, **solver_kwargs)
        c_fitter = constrained_fit(model_maker, **solver_kwargs)

        m, bonlypars = model_maker(params)
        exp_data = m.expected_data(bonlypars)
        # print(f'exp_data: {exp_data}')
        bounds = m.config.suggested_bounds()

        # map these
        initval = jnp.asarray([test_mu, 1.0])
        transforms = solver_kwargs.get("pdf_transform", False)
        if transforms:
            initval = to_inf_vec(initval, bounds)

        # the constrained fit

        # print('fitting constrained with init val %s setup %s', initval,[test_mu, nn_params])

        numerator = (
            to_bounded_vec(c_fitter(initval, [params, test_mu]), bounds)
            if transforms
            else c_fitter(initval, [params, test_mu])
        )

        # don't have to fit these -- we know them for expected limits!
        denominator = bonlypars  # to_bounded_vec(g_fitter(initval, params), bounds) if transforms else g_fitter(initval, params)

        # print(f"constrained fit: {numerator}")
        # print(f"global fit: {denominator}")

        # compute test statistic (lambda(µ))
        profile_likelihood = -2 * (
            m.logpdf(numerator, exp_data)[0] - m.logpdf(denominator, exp_data)[0]
        )

        # in exclusion fit zero out test stat if best fit µ^ is larger than test µ
        muhat = denominator[0]
        sqrtqmu = jnp.sqrt(jnp.where(muhat < test_mu, profile_likelihood, 0.0))
        return 1 - pyhf.tensorlib.normal_cdf(sqrtqmu)

    return pmu_jax


def CLs(model_maker, solver_kwargs):
    def get_CLs(params, test_mu):
        altval = 0
        CLsb = expected_pvalue_upper_limit(model_maker, solver_kwargs)(params, test_mu)
        CLb = 1 - pyhf.tensorlib.normal_cdf(altval)
        CLs = CLsb / CLb
        return CLs

    return get_CLs