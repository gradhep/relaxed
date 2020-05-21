__version__ = "0.0.1"

from .tensor import BackendRetriever as tensor
from .exceptions import InvalidBackend

tensorlib = tensor.jax_backend()
default_backend = tensorlib

# literally ripped straight from pyhf
def get_backend():
    """
    Get the current backend
    Example:
        >>> import smooth
        >>> smooth.get_backend()
        <smooth.tensor.numpy_backend.numpy_backend object at 0x...>

    Returns:
        backend
    """
    global tensorlib
    return tensorlib


def set_backend(backend):
    """
    Set the backend and the associated optimizer

    Example:
        >>> import smooth
        >>> smooth.set_backend("tensorflow")
        >>> smooth.tensorlib.name
        'tensorflow'
        >>> smooth.set_backend(b"pytorch")
        >>> smooth.tensorlib.name
        'pytorch'
        >>> smooth.set_backend(smooth.tensor.numpy_backend())
        >>> smooth.tensorlib.name
        'numpy'

    Args:
        backend (`str` or `smooth.tensor` backend): One of the supported smooth backends: NumPy, TensorFlow, PyTorch, and JAX

    Returns:
        None
    """
    global tensorlib
   

    if isinstance(backend, (str, bytes)):
        if isinstance(backend, bytes):
            backend = backend.decode("utf-8")
        backend = backend.lower()
        try:
            backend = getattr(tensor, "{0:s}_backend".format(backend))()
        except TypeError:
            raise InvalidBackend(
                "The backend provided is not supported: {0:s}. Select from one of the supported backends: numpy, tensorflow, pytorch".format(
                    backend
                )
            )

    _name_supported = getattr(tensor, "{0:s}_backend".format(backend.name))
    if _name_supported:
        if not isinstance(backend, _name_supported):
            raise AttributeError(
                "'{0:s}' is not a valid name attribute for backend type {1}\n                 Custom backends must have names unique from supported backends".format(
                    backend.name, type(backend)
                )
            )

    # need to determine if the tensorlib changed or the optimizer changed for events
    tensorlib_changed = bool(backend.name != tensorlib.name)

    # set new backend
    tensorlib = backend
    
    # trigger events
    if tensorlib_changed:
        events.trigger("tensorlib_changed")()

### diffable stuff ###
        
def hist(events, bins, bandwidth=None):
    """
    Args:
            events: (jax array-like) data to filter.
            
            bins: (jax array-like) intervals to calculate counts.
            
            bandwidth: (float) value that specifies the width of the individual
            distributions (kernels) whose cdfs are averaged over each bin.
    Returns:
            binned counts, calculated by kde!
    """

    #bandwidth = bandwidth or events.shape[-1]**-.25
    
    # grab bin edges
    edge_lo = bins[:-1]
    edge_hi = bins[1:]

    # get counts from gaussian cdfs centered on each event, evaluated binwise
    cdf_up = tensorlib.normal_cdf(edge_hi.reshape(-1, 1), mu=events, sigma=bandwidth)
    cdf_dn = tensorlib.normal_cdf(edge_lo.reshape(-1, 1), mu=events, sigma=bandwidth)
    summed = (cdf_up - cdf_dn).sum(axis=1)
    return summed


def cut(events, sign, cut_val, slope=1.0):
    """
    Event weights from cutting `events` at `cut_val` with logical operator `sign` = '>' or '<'.
    
    Chain cuts by multiplying their output: `evt_weights = cut(data1, sign1, c1) * cut(data2, sign2, c2) etc.    
    Args:
            events: (jax array-like) data to filter.
    Returns:
            event weights!
    """
    if sign == ">":
        passed = 1 / (1 + tensorlib.exp(-slope * (events - cut_val)))
    elif sign == "<":
        passed = 1 - (1 / (1 + tensorlib.exp(-slope * (events - cut_val))))
    else:
        print("Invalid cut sign -- use > or <.")

    return passed
