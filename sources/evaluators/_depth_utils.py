from typing import Final, Optional

import numba
import numpy as np
import numpy.typing as NP
from typing_extensions import Self

from ._abc import MetricAccumulator, MetricResult

ResultType = NP.NDArray[np.float64]


@numba.jit(nopython=True, nogil=True, cache=True)
def compute_metrics(true: NP.NDArray[np.float64], pred: NP.NDArray[np.float64]) -> NP.NDArray[np.float64]:
    eps = np.finfo(np.float32).eps
    true = np.maximum(true, eps)
    pred = np.maximum(pred, eps)

    # Error
    err = true - pred
    err_abs = np.abs(err)
    err_sq = err**2
    err_rel = err_abs / true

    # Inverse error
    true_inv = (1.0) / true
    pred_inv = (1.0) / pred
    err_inv_sq = (true_inv - pred_inv) ** 2

    # Inverse root mean squared error
    IRMSE = np.sqrt(np.mean(err_inv_sq))

    # Mean absolute error
    MAE = np.mean(err_abs)

    # Root mean squared error
    RMSE = np.sqrt(np.mean(err_sq))

    # Mean relative error
    ARE = np.mean(err_rel)

    # Root mean squared error
    err_rel_sq = err_sq / np.maximum(true**2, eps)
    RSE = np.mean(err_rel_sq)

    # Scale invariant logarithmic error
    err_log = np.log(true) - np.log(pred)
    n = len(pred)

    sile_1 = np.mean(err_log**2)
    sile_2 = (np.sum(err_log) ** 2) / (n**2)

    SILE = sile_1 - sile_2

    return np.array([IRMSE, MAE, RMSE, ARE, RSE, SILE])


class DepthMetrics(MetricResult):
    fields = ["IRMSE", "MAE", "RMSE", "ARE", "RSE", "SILE"]

    def __init__(
        self,
        categories: NP.NDArray[np.int64],
        true: ResultType,
        pred: ResultType,
    ):
        self._categories: Final = categories
        self._true = true
        self._pred = pred
        (
            self.IRMSE,
            self.MAE,
            self.RMSE,
            self.ARE,
            self.RSE,
            self.SILE,
        ) = compute_metrics(true, pred)

    def __len__(self) -> int:
        """
        Returns the amount of entries, i.e. the amount of valid pixels
        """
        return len(self._categories)

    def __getitem__(self, categories: NP.ArrayLike) -> Self:
        keep = np.isin(
            self._categories,
            np.asarray(categories),
        )

        return type(self)(
            categories=self._categories[keep],
            true=self._true[keep],
            pred=self._pred[keep],
        )


class DepthAccumulator(MetricAccumulator):
    """
    Implements depth evaluation metrics from the KITTI benchmark suite.
    """

    def __init__(self):
        self._true: list[ResultType] = []
        self._pred: list[ResultType] = []
        self._categories: list[NP.NDArray[np.int64]] = []

    def __len__(self):
        """
        Returns the amount of updates processed.
        """
        return len(self._categories)

    def reset(self):
        self.__init__()

    def update(
        self,
        *,
        depth_true: ResultType,
        depth_pred: ResultType,
        valid_mask: Optional[NP.NDArray[np.bool_]] = None,
        category_mask: Optional[NP.NDArray[np.int64]] = None,
    ):
        keep = depth_true > 0
        if valid_mask is not None:
            keep = keep * valid_mask

        depth_true = depth_true[keep].reshape(-1)
        depth_pred = depth_pred[keep].reshape(-1)

        assert len(depth_true) == len(depth_pred)

        if category_mask is None:
            categories = np.full_like(
                depth_true,
                fill_value=-1,
                dtype=np.int64,
            )
        else:
            categories = category_mask[keep].reshape(-1)

        self._true.append(depth_true.astype(np.float64))
        self._pred.append(depth_pred.astype(np.float64))
        self._categories.append(categories)

    def result(self) -> DepthMetrics:
        return DepthMetrics(
            categories=np.concatenate(self._categories),
            true=np.concatenate(self._true),
            pred=np.concatenate(self._pred),
        )

    def gather(self, other: Self) -> Self:
        if len(other) == 0:
            return self
        if len(self) == 0:
            return other

        # Extend self
        self._true.extend(other._true)
        self._pred.extend(other._pred)
        self._categories.extend(other._categories)

        # Clear other
        other._true.clear()
        other._pred.clear()
        other._categories.clear()

        return self
