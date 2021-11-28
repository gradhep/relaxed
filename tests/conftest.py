import jax.numpy as jnp
import pytest
from dummy_pyhf import uncorrelated_background


@pytest.fixture
def example_model():
    return uncorrelated_background(
        signal_data=jnp.asarray([5, 5]),
        bkg_data=jnp.asarray([50, 50]),
        bkg_uncerts=jnp.asarray([5, 5]),
    )
