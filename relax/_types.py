"""in part taken from deepmind/chex while waiting for a better way to type jax."""

__all__ = ["Array", "ArrayNumpy", "ArrayDevice", "PRNGKey"]
import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray
ArrayNumpy = np.ndarray
# Use this type for type annotation. For instance checking,  use
# `isinstance(x, jax.DeviceArray)`.
# `jax.interpreters.xla._DeviceArray` appears in jax > 0.2.5
if hasattr(jax.interpreters.xla, "_DeviceArray"):
    ArrayDevice = jax.interpreters.xla._DeviceArray
else:
    ArrayDevice = jax.interpreters.xla.DeviceArray
PRNGKey = Array
