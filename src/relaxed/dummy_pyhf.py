from __future__ import annotations

__all__ = ("example_model", "uncorrelated_background")

from typing import Any, Iterable

import jax.numpy as jnp
import jax
import pyhf
from equinox import Module as PyTree


class _Config(PyTree):
    poi_index: int
    npars: int
    auxdata: jax.Array
    def __init__(self, aux) -> None:
        self.poi_index = 0
        self.npars = 2
        self.auxdata = aux

    def suggested_init(self) -> jax.Array:
        return jnp.asarray([1.0, 1.0])
    
    def suggested_bounds(self) -> tuple[jax.Array, jax.Array]:
        return jnp.asarray([[0.0, 0.0], [10.0, 10.0]])


class Model(PyTree):
    """Dummy class to mimic the functionality of `pyhf.Model`."""
    sig: jax.Array
    nominal: jax.Array
    uncert: jax.Array
    factor: jax.Array
    config: _Config

    def __init__(self, spec: Iterable[Any]) -> None:
        self.sig, self.nominal, self.uncert = spec
        self.factor = (self.nominal / self.uncert) ** 2
        self.config = _Config(1.0 * self.factor)

    def expected_data(self, pars: jax.Array) -> jax.Array:
        mu, gamma = pars
        expected_main = jnp.asarray([gamma * self.nominal + mu * self.sig])
        return jnp.concatenate([expected_main, jnp.array([self.config.auxdata])])

    # logpdf as the call method
    def logpdf(self, pars: jax.Array, data: jax.Array) -> jax.Array:
        maindata, auxdata = data
        main, _ = self.expected_data(pars)
        _, gamma = pars
        main = pyhf.probability.Poisson(main).log_prob(maindata)
        constraint = pyhf.probability.Poisson(gamma * self.factor).log_prob(auxdata)
        # sum log probs over bins
        return [jnp.sum(jnp.asarray([main + constraint]), axis=None)]


def uncorrelated_background(s: jax.Array, b: jax.Array, db: jax.Array) -> Model:
    """Dummy class to mimic the functionality of `pyhf.simplemodels.hepdata_like`."""
    return Model([s, b, db])


def _calc_yields(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s = 15 + x
    b = 45 - 2 * x
    db = 1 + 0.2 * x**2
    return [s], [b], [db]


def example_model(
    phi: jnp.ndarray, return_yields: bool = False
) -> Model | tuple[Model, jnp.ndarray]:
    s, b, db = yields = _calc_yields(phi)

    model = uncorrelated_background(
        jnp.asarray([s]), jnp.asarray([b]), jnp.asarray([db])
    )

    if return_yields:
        return model, yields
    else:
        return model
