import pytest
from dummy_pyhf import uncorrelated_background


@pytest.fixture
def example_model():
    return uncorrelated_background(5, 50, 5)
