from functools import partial

import jax.numpy as jnp
import numpy as np
import pytest
from jax import jacrev, vmap
from jax.random import PRNGKey, normal

import relaxed


@pytest.fixture
def big_sample():
    return normal(PRNGKey(0), shape=(5000,))


@pytest.fixture
def bins():
    return np.linspace(-5, 5, 10)


def test_hist_validity(big_sample, bins):
    numpy_hist = np.histogram(big_sample, bins=bins)[0]
    relaxed_hist = relaxed.hist(big_sample, bins=bins, bandwidth=1e-6)
    assert np.allclose(numpy_hist, relaxed_hist)


def test_hist_validity_density(big_sample, bins):
    numpy_hist = np.histogram(big_sample, bins=bins, density=True)[0]
    relaxed_hist = relaxed.hist(big_sample, bins=bins, bandwidth=1e-6, density=True)
    assert np.allclose(numpy_hist, relaxed_hist)


def test_hist_approx_validity(big_sample, bins):
    """Roughly test validity of hist for wider bandwidth (and a more loose criterion).
    Useful because it's the same bandwidth as the gradient test below."""
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
        relaxed_grads, grads, atol=0.01
    )  # atol quite tight, 0.001 will fail (outliers)


# def test_fisher_info():
