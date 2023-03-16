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


def stable_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Divide two arrays element-wise, returning zero when the divisor is zero.
    """

    return np.true_divide(a, b, out=np.zeros_like(a), where=b != 0)
