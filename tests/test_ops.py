from functools import partial

import jax.numpy as jnp
import numpy as np
import pyhf
import pytest
from dummy_pyhf import example_model
from jax import jacrev, vmap
from jax.random import PRNGKey, normal

import relaxed


@pytest.fixture
def big_sample():
    return normal(PRNGKey(1), shape=(5000,))


@pytest.fixture
def bins():
    return np.linspace(-5, 5, 6)


def test_hist_validity(big_sample, bins):
    numpy_hist = np.histogram(big_sample, bins=bins)[0]
    relaxed_hist = relaxed.hist(big_sample, bins=bins, bandwidth=1e-6)
    assert np.allclose(numpy_hist, relaxed_hist)


def test_hist_validity_density(big_sample, bins):
    numpy_hist = np.histogram(big_sample, bins=bins, density=True)[0]
    relaxed_hist = relaxed.hist(big_sample, bins=bins, bandwidth=1e-6, density=True)
    assert np.allclose(numpy_hist, relaxed_hist)


def test_hist_validity_infinities(big_sample, bins):
    """Test the reflection of excess density @ [-inf,inf] into the edge bins."""
    inf_bins = [-np.inf, *bins, np.inf]
    numpy_hist = np.histogram(big_sample, bins=inf_bins, density=True)[0]
    numpy_hist[1] += numpy_hist[0]
    numpy_hist[-2] += numpy_hist[-1]
    numpy_hist = numpy_hist[1:-1]
    relaxed_hist = relaxed.hist(
        big_sample, bins=bins, bandwidth=1e-6, density=True, reflect_infinities=True
    )
    assert np.allclose(numpy_hist, relaxed_hist)


def test_hist_approx_validity(big_sample, bins):
    """Roughly test validity of hist for wider bandwidth (and a more loose criterion).
    Useful because it's the same bandwidth (0.15) as the gradient test below."""
    numpy_hist = np.histogram(big_sample, bins=bins, density=True)[0]
    relaxed_hist = relaxed.hist(big_sample, bins=bins, bandwidth=0.15, density=True)
    assert np.allclose(numpy_hist, relaxed_hist, atol=0.01)


def test_hist_grad_validity(bins):
    """Test the grads of the kde hist vs the analyitc grads of a normal dist wrt mu."""

    def gen_points(mu, jrng, nsamples):
        points = normal(jrng, shape=(nsamples,)) + mu
        return points

    def bin_height(mu, jrng, bw, nsamples, bins):
        points = gen_points(mu, jrng, nsamples)
        return relaxed.hist(points, bins, bandwidth=bw)

    mus = jnp.linspace(-2, 2, 100)

    def kde_grads(bw, nsamples):
        rngs = [PRNGKey(i) for i in range(5)]
        grad_fun = jacrev(bin_height)
        grads = []
        for jrng in rngs:
            get_grads = vmap(
                partial(grad_fun, jrng=jrng, bw=bw, nsamples=nsamples, bins=bins)
            )
            grads.append(get_grads(mus))
        return jnp.asarray(grads)

    nsamples = 5000
    relaxed_grads = kde_grads(bw=0.15, nsamples=nsamples).mean(axis=0) / nsamples

    def true_grad(mu, bins):
        """Analytic grad of the mean height over an interval of a normal dist wrt mu."""
        # The full equation in latex for copy/paste (delete the # statement at the end):
        # \frac{\partial}{\partial\mu}bin_{\mathsf{true}}(\mu) = -\frac{1}{\sqrt{2\\pi}}\left[\left(e^{-\frac{(b-\mu)^2}{2}}\right) - \left( e^{-\frac{(a-\mu)^2}{2}}\right)\right] # fmt: skip
        b = bins[1:]  # ending bin edges ||<-
        a = bins[:-1]  # starting bin edges ->||
        return -(1 / ((2 * jnp.pi) ** 0.5)) * (
            jnp.exp(-((b - mu) ** 2) / 2) - jnp.exp(-((a - mu) ** 2) / 2)
        )

    grads = vmap(partial(true_grad, bins=bins))(mus)

    assert np.allclose(
        relaxed_grads, grads, atol=0.01, rtol=0.05
    )  # tols are a bit high because the grads are a little noisy/biased


def test_fisher_info():
    model = example_model(5.0)
    pars = model.config.suggested_init()
    data = model.expected_data(pars)
    # just check that it doesn't crash
    relaxed.fisher_info(model, pars, data)


def test_fisher_uncerts_validity():
    pyhf.set_backend("jax", pyhf.optimize.minuit_optimizer(verbose=1))

    m = pyhf.simplemodels.uncorrelated_background([5, 5], [50, 50], [5, 5])

    data = jnp.array([50.0, 50.0] + m.config.auxdata)

    fit_res = pyhf.infer.mle.fit(
        data,
        m,
        return_uncertainties=True,
        par_bounds=[
            [-1, 10],
            [-1, 10],
            [-1, 10],
        ],  # fit @ boundary produces unstable uncerts
    )

    # minuit fit uncerts
    mle_pars, mle_uncerts = fit_res[:, 0], fit_res[:, 1]

    # uncertainties from autodiff hessian
    relaxed_uncerts = relaxed.cramer_rao_uncert(m, mle_pars, data)
    assert np.allclose(mle_uncerts, relaxed_uncerts, rtol=5e-2)


def test_fisher_info_grad():
    def pipeline(x):
        model = example_model(5.0)
        pars = model.config.suggested_init()
        data = model.expected_data(pars)
        return relaxed.fisher_info(model, pars * x, data * x)

    jacrev(pipeline)(4.0)  # just check you can calc it w/o exception


def test_fisher_uncert_grad():
    def pipeline(x):
        model = example_model(5.0)
        pars = model.config.suggested_init()
        data = model.expected_data(pars)
        return relaxed.cramer_rao_uncert(model, pars * x, data * x)

    jacrev(pipeline)(4.0)  # just check you can calc it w/o exception


@pytest.mark.parametrize("keep", ["above", "below"])
def test_cut_validity(big_sample, keep):
    """Recover hard cut in large slope case."""
    weights = relaxed.cut(big_sample, 0, slope=100000, keep=keep)
    if keep == "above":
        np_cut = np.array(big_sample > 0, dtype=float)
    elif keep == "below":
        np_cut = np.array(big_sample < 0, dtype=float)

    assert np.allclose(weights, np_cut)


@pytest.mark.parametrize("keep", ["above", "below"])
def test_cut_grad(keep):
    def pipeline(x):
        model = example_model(5.0)
        pars = model.config.suggested_init()
        data = model.expected_data(pars)
        return relaxed.cut(data, x, keep=keep)

    jacrev(pipeline)(4.0)  # just check you can calc it w/o exception


def test_invalid_cut_keep():
    with pytest.raises(ValueError):
        relaxed.cut(jnp.array([1, 2, 3]), 0, keep="frogs")
