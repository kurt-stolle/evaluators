"""
Monocular Depth Estimation Evaluators
"""

from collections import defaultdict
from functools import cached_property, reduce
from typing import Final, Iterable, Optional

import jax
import jax.numpy as jnp
import numpy.typing as NP
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from tabulate import tabulate
from torch import Tensor
from typing_extensions import Self

from ._abc import MetricAccumulator, MetricResult
from ._utils import stable_div, stable_inv_difference
from .base_evaluator import BaseEvaluator

__all__ = ["DepthEvaluator"]

ResultType = jnp.ndarray


_logger = setup_logger(name=f"{__name__}_{comm.get_local_rank()}")


@jax.jit
def compute_metrics(true, pred):
    # Error
    err = true - pred
    err_abs = jnp.abs(err)
    err_sq = err**2
    err_rel = stable_div(err_abs, true)

    # Inverse error
    err_inv_sq = stable_inv_difference(pred, true) ** 2

    # Inverse root mean squared error
    IRMSE = jnp.sqrt(jnp.mean(err_inv_sq))

    # Mean absolute error
    MAE = jnp.mean(err_abs)

    # Root mean squared error
    RMSE = jnp.sqrt(jnp.mean(err_sq))

    # Mean relative error
    ARE = jnp.mean(err_rel)

    # Root mean squared error
    true_sq = true**2
    err_rel_sq = jnp.where(true_sq > 0, jnp.true_divide(err_sq, true_sq), jnp.zeros_like(err_sq))
    RSE = jnp.mean(err_rel_sq)

    # Scale invariant logarithmic error
    err_log = jnp.log(true) - jnp.log(pred)
    err_log = jnp.where(jnp.isfinite(err_log), err_log, jnp.zeros_like(err_log))
    n = jnp.float32(max(1, len(pred)))

    sile_1 = jnp.mean(err_log**2)
    sile_2 = jnp.sum(err_log) ** 2
    sile_2 /= n
    sile_2 /= n

    SILE = sile_1 - sile_2

    assert IRMSE.shape == MAE.shape
    assert IRMSE.shape == RMSE.shape
    assert IRMSE.shape == ARE.shape
    assert IRMSE.shape == RSE.shape
    assert IRMSE.shape == SILE.shape

    return jnp.array([IRMSE, MAE, RMSE, ARE, RSE, SILE])


class DepthMetrics(MetricResult):
    fields = ["IRMSE", "MAE", "RMSE", "ARE", "RSE", "SILE"]

    def __init__(
        self,
        categories: jnp.ndarray,
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
        keep = jnp.isin(
            self._categories,
            jnp.asarray(categories),
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
        self._categories: list[jnp.ndarray] = []

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
        valid_mask: Optional[jnp.ndarray] = None,
        category_mask: Optional[jnp.ndarray] = None,
    ):
        keep = depth_true > 0
        if valid_mask is not None:
            keep = keep * valid_mask

        depth_true = depth_true[keep].reshape(-1)
        depth_pred = depth_pred[keep].reshape(-1)

        assert len(depth_true) == len(depth_pred)

        if category_mask is None:
            categories = jnp.full_like(
                depth_true,
                fill_value=-1,
                dtype=jnp.int32,
            )
        else:
            categories = category_mask[keep].reshape(-1)

        self._true.append(depth_true.astype(jnp.float64))
        self._pred.append(depth_pred.astype(jnp.float64))
        self._categories.append(categories)

    def result(self) -> DepthMetrics:
        return DepthMetrics(
            categories=jnp.concatenate(self._categories),
            true=jnp.concatenate(self._true),
            pred=jnp.concatenate(self._pred),
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


class DepthEvaluator(BaseEvaluator):
    task_name = "task_depth"

    def __init__(
        self,
        *,
        ignored_label: int,
        label_divisor: int,
        thing_classes: Iterable[int],
        stuff_classes: Iterable[int],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ignored_label: Final = ignored_label
        self.label_divisor: Final = label_divisor
        self.thing_classes: Final = list(thing_classes)
        self.stuff_classes: Final = list(_id for _id in stuff_classes if _id not in self.thing_classes)
        self.metrics = DepthAccumulator()

    @classmethod
    def from_metadata(cls, dataset_name: str, **kwargs) -> Self:
        from detectron2.data import MetadataCatalog

        m = MetadataCatalog.get(dataset_name)

        thing_classes = list(m.thing_translations.values())
        stuff_classes = list(_id for _id in m.stuff_translations.values() if _id not in thing_classes)

        return cls(
            ignored_label=m.ignore_label,
            label_divisor=m.label_divisor,
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
            **kwargs,
        )

    @cached_property
    def num_classes(self):
        return len(self.stuff_classes) + len(self.thing_classes)

    def reset(self):
        self.metrics = DepthAccumulator()

    def process_item(
        self,
        input_: dict[str, Tensor],
        output: dict[str, Tensor],
    ):
        depth_true = input_.get("depth")
        if depth_true is None or depth_true.max() == 0.0:
            return
        depth_true = depth_true.detach().cpu()

        depth_pred = output.get("depth")
        if depth_pred is None:
            raise ValueError("Output has no estimated depth map")
        depth_pred = depth_pred.detach().cpu()

        sem_seg = input_.get("sem_seg")
        if sem_seg is not None:
            valid_mask = (sem_seg != self.ignored_label).detach().cpu().numpy()
            sem_seg = sem_seg.detach().cpu().numpy()
        else:
            valid_mask = None

        self.metrics.update(
            depth_true=depth_true.numpy(),
            depth_pred=depth_pred.numpy(),
            valid_mask=valid_mask,
            category_mask=sem_seg,
        )

    def evaluate(self) -> Optional[dict[str, dict[str, float]]]:
        comm.synchronize()
        self.metrics = reduce(lambda a, b: a and a.gather(b) or b, comm.gather(self.metrics), None)

        if not comm.is_main_process():
            return None

        _logger.info("Combining depth evaluator results...")

        result = self.metrics.result()
        output = {}

        _logger.info("Computing results for overall, things and stuff...")

        for field, value in result:
            output[field] = value

        for field, value in result[self.thing_classes]:
            output[field + "_th"] = value

        for field, value in result[self.stuff_classes]:
            output[field + "_st"] = value

        self.print_results(result)

        return {self.task_name: output}

    def print_results(self, dm: DepthMetrics):
        data = defaultdict(list)
        data[""].append("All")
        for field, value in dm:
            data[field].append(value)

        data[""].append("Things")
        for field, value in dm[self.thing_classes]:
            data[field].append(value)

        data[""].append("Stuff")
        for field, value in dm[self.stuff_classes]:
            data[field].append(value)

        table = tabulate(
            list(zip(*data.values())),
            headers=list(data.keys()),
            tablefmt="pipe",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )
        _logger.info("Depth evaluation results:\n" + table)
