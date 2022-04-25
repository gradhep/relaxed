from __future__ import annotations

__all__ = ("example_model", "uncorrelated_background")

from typing import Any, Iterable

import jax.numpy as jnp
import pyhf


# class-based
class _Config:
    def __init__(self) -> None:
        self.poi_index = 0
        self.npars = 2

    def suggested_init(self) -> jnp.ndarray:
        return jnp.asarray([1.0, 1.0])


class Model:
    """Dummy class to mimic the functionality of `pyhf.Model`."""

    def __init__(self, spec: Iterable[Any]) -> None:
        pyhf.set_backend("jax")
        self.sig, self.nominal, self.uncert = spec
        self.factor = (self.nominal / self.uncert) ** 2
        self.aux = 1.0 * self.factor
        self.config = _Config()

    def expected_data(self, pars: jnp.ndarray) -> jnp.ndarray:
        mu, gamma = pars
        expected_main = jnp.asarray([gamma * self.nominal + mu * self.sig])
        aux_data = jnp.asarray([self.aux])
        return jnp.concatenate([expected_main, aux_data])

    def logpdf(self, pars: jnp.ndarray, data: jnp.ndarray) -> jnp.ndarray:
        maindata, auxdata = data
        main, _ = self.expected_data(pars)
        _, gamma = pars
        main = pyhf.probability.Poisson(main).log_prob(maindata)
        constraint = pyhf.probability.Poisson(gamma * self.factor).log_prob(auxdata)
        # sum log probs over bins
        return [jnp.sum(jnp.asarray([main + constraint]), axis=None)]


def uncorrelated_background(s: jnp.ndarray, b: jnp.ndarray, db: jnp.ndarray) -> Model:
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
