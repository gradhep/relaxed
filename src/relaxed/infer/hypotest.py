"""Calculate expected CLs values with hypothesis tests."""
from __future__ import annotations

__all__ = ["make_hypotest"]

from typing import Any, Callable

import jax
import jax.numpy as jnp
import pyhf

from .._types import ArrayDevice
from ..fit import constrained_fit
from ..fit.minuit_transforms import to_bounded_vec, to_inf_vec

pyhf.set_backend("jax")
jax.config.update("jax_enable_x64", True)


def make_hypotest(
    model_maker: Callable[..., tuple[Any, ArrayDevice]],
    solver_kwargs: dict[str, Any] = dict(pdf_transform=True),
) -> Callable[[ArrayDevice, float, list[str]], dict[str, float]]:
    """Instatiate a hypotheses test based on a model maker.

    Parameters
    ----------
    model_maker: function that takes model params and returns the tuple
    [model, background-only parameters] when called.
    solver_kwargs: keyword arguments passed to constrained_fit.

    Returns
    -------
    hypotest: function that performs a hypothesis test using the profile
    likelihood as a test statistic. calculates CLs values in the expected
    case of observing the nominal signal + background yields.
    """

    @jax.jit
    def hypotest(
        hyperpars: ArrayDevice,
        test_mu: float,
        metrics: list[str] = ["CLs"],
        model_kwargs: dict[str, Any] = dict(),
    ) -> dict[str, float]:
        # g_fitter = global_fit(model_maker, **solver_kwargs)
        c_fitter = constrained_fit(model_maker, model_kwargs, **solver_kwargs)

        m, bonlypars = model_maker(hyperpars, **model_kwargs)
        exp_data = m.expected_data(bonlypars)
        bounds = jnp.array(m.config.suggested_bounds())
        # map these
        initval = jnp.asarray([test_mu, 1.0])
        transforms = solver_kwargs.get("pdf_transform", False)
        if transforms:
            initval = to_inf_vec(initval, bounds)

        # the constrained fit
        numerator = (
            to_bounded_vec(c_fitter(initval, (hyperpars, test_mu)), bounds)
            if transforms
            else c_fitter(initval, (hyperpars, test_mu))
        )

        # don't have to fit these -- we know them for expected limits!
        denominator = bonlypars

        # compute test statistic (lambda(µ))
        profile_likelihood = -2 * (
            m.logpdf(numerator, exp_data)[0] - m.logpdf(denominator, exp_data)[0]
        )

        # in exclusion fit zero out test stat if best fit µ^ is larger than test µ
        muhat = denominator[0]
        sqrtqmu = jnp.sqrt(jnp.where(muhat < test_mu, profile_likelihood, 0.0))
        CLsb = 1 - pyhf.tensorlib.normal_cdf(sqrtqmu)
        altval = 0
        CLb = 1 - pyhf.tensorlib.normal_cdf(altval)
        CLs = CLsb / CLb

        pdict = dict(CLs=CLs, p_sb=CLsb, p_b=CLb, profile_likelihood=profile_likelihood)
        return {k: pdict[k] for k in metrics}

    return hypotest
