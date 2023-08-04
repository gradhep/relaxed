import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import Array
from typing import Sequence

jax.config.update("jax_enable_x64", True)


@jax.jit
def poisson_logpdf(n, lam):
    return n * jnp.log(lam) - lam - jsp.special.gammaln(n + 1)


class Model(eqx.Module):
    def logpdf(self, data: Array, pars: dict[str, Array] | Array) -> Array:
        raise NotImplementedError

    def expected_data(self, pars: dict[str, Array] | Array) -> Array:
        raise NotImplementedError


class Systematic(eqx.Module):
    name: str = eqx.field(static=True)
    constraint: Model


class PoissonConstraint(Model):
    scaled_binwise_uncerts: Array

    def __init__(self, nominal_bkg: Array, binwise_uncerts: Array) -> None:
        if nominal_bkg.shape != binwise_uncerts.shape:
            msg = f"Nominal bkg shape {nominal_bkg.shape} does not match binwise uncertainty shape {binwise_uncerts.shape}"
            raise ValueError(msg)
        self.scaled_binwise_uncerts = binwise_uncerts / nominal_bkg

    def expected_data(self, gamma: Array) -> Array:
        return gamma * self.scaled_binwise_uncerts**-2

    def logpdf(self, auxdata, gamma):
        if not isinstance(gamma, Array):
            gamma = jnp.array(gamma)
        if gamma.shape != self.scaled_binwise_uncerts.shape and not (
            gamma.shape == () and self.scaled_binwise_uncerts.shape == (1,)
        ):
            msg = f"Constrained param shape {gamma.shape} does not match number of bins {self.scaled_binwise_uncerts.shape}"
            raise ValueError(msg)
        return jnp.sum(
            poisson_logpdf(auxdata, (gamma * self.scaled_binwise_uncerts**-2)),
            axis=None,
        )


class UncorrelatedShape(Systematic):
    def __init__(self, name: str, nominal_bkg: Array, binwise_uncerts: Array) -> None:
        self.name = name
        self.constraint = PoissonConstraint(nominal_bkg, binwise_uncerts)


class HEPDataLike(Model):
    sig: Array
    bkg: Array
    db: Array
    systematic: UncorrelatedShape

    def __init__(self, sig: Array, bkg: Array, db: Array) -> None:
        self.sig = sig
        self.bkg = bkg
        self.db = db
        self.systematic = UncorrelatedShape("shapesys", bkg, db)

    @property
    def auxdata(self) -> Array:
        return self.systematic.constraint.expected_data(jnp.ones_like(self.db))

    def expected_data(self, pars: dict[str, Array]) -> Array:
        mu, gamma = pars["mu"], pars["shapesys"]
        return (
            mu * self.sig + gamma * self.bkg,
            self.systematic.constraint.expected_data(gamma),
        )

    def logpdf(self, data: Array, pars: dict[str, Array]) -> Array:
        # check data shape
        if not isinstance(data, Sequence) or len(data) != 2:
            msg = f"Data should be a tuple of (maindata, auxdata), got {data}"
            raise ValueError(msg)
        # check size of maindata
        maindata = data[0]
        if maindata.shape != self.sig.shape:
            msg = f"Main data shape {maindata.shape} does not match number of signal bins {self.sig.shape}"
            raise ValueError(msg)
        auxdata = data[1]
        main, _ = self.expected_data(pars)
        main = jnp.sum(poisson_logpdf(maindata, main), axis=None)
        constraint = self.systematic.constraint.logpdf(auxdata, pars["shapesys"])
        return main + constraint
