import jax
import jax.numpy as jnp
import numpy as np


def count_labels(id_array: np.ndarray) -> dict[np.uint64, np.uint64]:
    """
    Count the number of occurrences of each label in an array.

    Parameters
    ----------
    id_array : np.ndarray
        The array of labels.

    Returns
    -------
    dict[np.uint64, np.uint64]
        A dictionary mapping labels to counts.
    """

    ids, counts = np.unique(id_array, return_counts=True)
    return dict(zip(ids, counts.astype(np.uint64)))


@jax.jit
def stable_div(a, b):
    """
    Divide two arrays element-wise, returning zero when the divisor is zero.
    """

    mask = b != 0
    div = jnp.where(mask, jnp.divide(a, b), jnp.zeros_like(a))
    return div


@jax.jit
def stable_inv_difference(a, b):
    """
    Returns the inverse difference where neither value is zero.
    """

    mask = (b != 0) & (a != 0)

    return jnp.where(mask, 1.0 / a - 1.0 / b, jnp.zeros_like(a))
