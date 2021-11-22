import jax
import numpy as np
import pytest
from jax.random import PRNGKey, normal

import relaxed


@pytest.fixture
def big_sample():
    return normal(PRNGKey(0), shape=(1000,))


def test_hist_validity(big_sample):
    bins = np.linspace(-5, 5, 10)
    numpy_hist = np.histogram(big_sample, bins=bins)[0]
    jax_hist = relaxed.hist(big_sample, bins=bins, bandwidth=0.0001)
    assert np.allclose(numpy_hist, jax_hist)


def test_hist_grad(big_sample):
    def modified_sample(x):
        data = big_sample * x
        return relaxed.hist(data, bins=np.linspace(-5, 5, 10), bandwidth=0.01)[1]

    assert jax.value_and_grad(modified_sample)(
        7.0
    )  # [1] == 1.0 # TODO: what should it be?
