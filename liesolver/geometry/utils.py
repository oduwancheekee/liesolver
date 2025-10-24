import numpy as np
from functools import wraps


def isclose(a, b):
    """A modified version of `np.isclose` for DeepXDE.

    This function changes the value of `atol` due to the dtype of `a` and `b`.
    If the dtype is float16, `atol` is `1e-4`.
    If it is float32, `atol` is `1e-6`.
    Otherwise (for float64), the default is `1e-8`.
    If you want to manually set `atol` for some reason, use `np.isclose` instead.

    Args:
        a, b (array like): Input arrays to compare.
    """
    a_dtype = np.asarray(a).dtype
    b_dtype = np.asarray(b).dtype
    atol = 1e-8
    if np.float32 in [a_dtype, b_dtype]:
        atol = 1e-6
    if np.float16 in [a_dtype, b_dtype]:
        atol = 1e-4
    return np.isclose(a, b, atol=atol)


def vectorize(**kwargs):
    """numpy.vectorize wrapper that works with instance methods.

    References:

    - https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
    - https://stackoverflow.com/questions/48981501/is-it-possible-to-numpy-vectorize-an-instance-method
    - https://github.com/numpy/numpy/issues/9477
    """

    def decorator(fn):
        vectorized = np.vectorize(fn, **kwargs)

        @wraps(fn)
        def wrapper(*args):
            return vectorized(*args)

        return wrapper

    return decorator
