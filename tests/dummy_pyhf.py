import jax.scipy as jsp
import equinox as eqx
import jax.numpy as jnp
from jax import Array
import jax

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
        eqx.error_if(
            nominal_bkg, 
            nominal_bkg.shape != binwise_uncerts.shape, 
            f"Nominal bkg shape {nominal_bkg.shape} does not match binwise uncertainty shape {binwise_uncerts.shape}"
        )
        self.scaled_binwise_uncerts = binwise_uncerts / nominal_bkg
    
    def expected_data(self, gamma: Array) -> Array:
        return gamma*self.scaled_binwise_uncerts**-2
    
    def logpdf(self, auxdata, gamma):
        eqx.error_if(
            gamma, 
            gamma.shape != self.scaled_binwise_uncerts.shape, 
            f"Constrained param shape {gamma.shape} does not match number of bins {self.scaled_binwise_uncerts.shape}"
        )
        return jnp.sum(
            poisson_logpdf(auxdata, (gamma*self.scaled_binwise_uncerts**-2)), 
            axis=None
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

    def expected_data(self, pars: dict[str, Array]) -> Array:
        mu, gamma = pars["mu"], pars["shapesys"]
        return mu * self.sig + gamma * self.bkg, self.systematic.constraint.expected_data(gamma)
    
    def logpdf(self, data: Array, pars: dict[str, Array]) -> Array:
        maindata, auxdata = data
        main, _ = self.expected_data(pars)
        main = jnp.sum(poisson_logpdf(maindata, main), axis=None)
        constraint = self.systematic.constraint.logpdf(auxdata, pars["shapesys"])
        return main + constraint



def _calc_yields(x: jnp.ndarray, n_bins: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    s = 15 + x
    b = 45 - 2 * x
    db = 1 + 0.2 * x**2
    return [s]*n_bins, [b]*n_bins, [db]*n_bins


def example_model(
    phi: jnp.ndarray, n_bins:int, return_yields: bool = False
) -> Model | tuple[Model, jnp.ndarray]:
    s, b, db = yields = _calc_yields(phi, n_bins)

    model = HEPDataLike(
        jnp.asarray(s), jnp.asarray(b), jnp.asarray(db)
    )

    if return_yields:
        return model, yields
    return model
