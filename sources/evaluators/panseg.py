"""
Panoptic Segmentation Evaluator
"""

from __future__ import annotations

import itertools
import multiprocessing as mp
from functools import cached_property, partial
from logging import warn
from typing import Any, Final, NamedTuple, Optional

import numpy as np
import numpy.typing as NP
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from tabulate import tabulate
from torch import Tensor

from ._shm import EvalPair
from ._types import Exposures, Outcomes
from ._utils import count_labels, stable_div
from .base_evaluator import BaseEvaluator

__all__ = ["PanSegEvaluator"]

_logger = setup_logger(name=__name__)


class PQResult(NamedTuple):
    """
    Resulting PQ, RQ and SQ as a C-dimensional array, where C is the amount of
    classes.
    """

    PQ: NP.NDArray[np.float64]
    RQ: NP.NDArray[np.float64]
    SQ: NP.NDArray[np.float64]


class PQStat(NamedTuple):
    """
    Resulting PQ, RQ and SQ as a C-dimensional array, where C is the amount of
    classes.
    """

    iou: NP.NDArray[np.float64]
    tp: NP.NDArray[np.float64]
    fp: NP.NDArray[np.float64]
    fn: NP.NDArray[np.float64]


class PanSegEvaluator(BaseEvaluator):
    """
    Wrapper around PQ for Detectron2 evaluation.
    """

    task_name = "task_panseg"

    def __init__(
        self,
        *,
        thing_classes: list[int],
        stuff_classes: list[int],
        label_divisor: int,
        ignore_label: int,
        thing_names: Optional[list[str]] = None,
        stuff_names: Optional[list[str]] = None,
        mask_key: Optional[str] = None,
        offset=2**20,
    ):
        self.label_divisor: Final = label_divisor

        if thing_names is None:
            thing_names = [str(i) for i in thing_classes]

        if stuff_names is None:
            stuff_names = [str(i) for i in stuff_classes]

        self.thing_classes = thing_classes
        self.thing_names = thing_names
        self.stuff_classes, self.stuff_names = map(
            list, zip(*[(id_, name) for id_, name in zip(stuff_classes, stuff_names) if id_ not in self.thing_classes])
        )

        self.mask_key = mask_key

        self.ignore_label: Final = ignore_label
        self.offset: Final = offset

        # Items storage
        self._items: list[EvalPair] = []

    @classmethod
    def from_metadata(cls, dataset_name: str, **kwargs) -> PanSegEvaluator:
        metadata = MetadataCatalog.get(dataset_name)

        return cls(
            label_divisor=metadata.label_divisor,
            thing_classes=list(metadata.thing_translations.values()),
            thing_names=list(metadata.thing_classes),
            stuff_classes=list(metadata.stuff_translations.values()),
            stuff_names=list(metadata.stuff_classes),
            ignore_label=metadata.ignore_label,
            **kwargs,
        )

    @cached_property
    def num_classes(self):
        return len(self.stuff_classes) + len(self.thing_classes)

    def reset(self):
        self._items = []

    def process_item(self, input_: dict[str, Any], output: dict[str, Any]) -> None:
        # Check if sample has annotations
        true = input_["labels"]

        # Read output
        pred, _ = output.get("panoptic_seg", (None, None))
        if pred is None:
            raise ValueError("No panoptic prediction!")

        pred = pred.clone()
        pred[pred == -1] = self.ignore_label

        # Masked predictions
        if self.mask_key is not None:
            mask: Tensor = input_[self.mask_key] > 0
            pred[mask] = self.ignore_label * self.label_divisor
            true[mask] = self.ignore_label * self.label_divisor

        self._items.append(EvalPair(true=true.cpu().numpy(), pred=pred.cpu().numpy()))

    def evaluate(self):
        comm.synchronize()

        p_count = mp.cpu_count() // comm.get_world_size()
        with mp.Pool(p_count) as p:
            pqs = p.starmap(
                accumulate,
                (
                    (
                        item.true,
                        item.pred,
                        self.num_classes,
                        self.ignore_label,
                        self.label_divisor,
                        self.offset,
                    )
                    for item in self._items
                ),
            )
        p.join()
        self.reset()

        pqs = list(itertools.chain(*comm.gather(pqs)))  # type: ignore
        if not comm.is_main_process():
            return

        assert len(pqs) > 0
        assert len(pqs[0]) == 4

        pqs_total = PQStat(*map(partial(np.sum, axis=0), zip(*pqs)))  # type: ignore
        res = evaluate_pq(
            iou=pqs_total.iou,
            tp=pqs_total.tp,
            fp=pqs_total.fp,
            fn=pqs_total.fn,
        )
        pqs_mean = PQStat(*map(partial(np.mean, axis=0), zip(*pqs)))  # type: ignore

        output = {}
        for key, values in {
            "PQ": res.PQ,
            "SQ": res.SQ,
            "RQ": res.RQ,
        }.items():
            output[key] = values.mean()
            output[key + "_th"] = values[self.thing_classes].mean()
            output[key + "_st"] = values[self.stuff_classes].mean()

        self.print_result(pqs_mean, res)

        return {self.task_name: output}

    def print_result(self, pqs: PQStat, res: PQResult, per_category=True):
        cats_th = self.thing_classes
        cats_st = self.stuff_classes
        cats = cats_th + cats_st

        names_th = self.thing_names
        names_st = self.stuff_names
        names = names_th + names_st

        data = []
        for label, select in itertools.chain(
            [("All", cats), ("Things", cats_th), ("Stuff", cats_st)],
            zip(names, cats),
        ):
            label = label.capitalize()

            if isinstance(select, int):
                num = "-"
            else:
                num = str(len(select))

            data.append(
                [
                    label,
                    res.PQ[select].mean(),
                    res.SQ[select].mean(),
                    res.RQ[select].mean(),
                    num,
                    pqs.iou[select].mean(),
                    pqs.tp[select].mean(),
                    pqs.fn[select].mean(),
                    pqs.fp[select].mean(),
                ]
            )

        table = tabulate(
            data,
            headers=[
                "",
                "PQ",
                "SQ",
                "RQ",
                "#",
                "IoU",
                "TP",
                "FN",
                "FP",
            ],
            tablefmt="pipe",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )
        _logger.info("PQ evaluation results:\n" + table)


def evaluate_pq(
    iou: NP.NDArray[np.float64],
    tp: NP.NDArray[np.float64],
    fp: NP.NDArray[np.float64],
    fn: NP.NDArray[np.float64],
) -> PQResult:
    sq = stable_div(iou, tp)
    rq = stable_div(tp, tp + 0.5 * fp + 0.5 * fn)
    pq = sq * rq

    # Mask out metrics that have a sum total of 0 for TP, FN and FP
    mask = np.equal(tp + fn + fp, 0)

    _logger.info(f"Invalid entries: {mask.sum()}")

    # Return results
    return PQResult(
        PQ=np.ma.MaskedArray(pq, mask) * 100,
        SQ=np.ma.MaskedArray(sq, mask) * 100,
        RQ=np.ma.MaskedArray(rq, mask) * 100,
    )


def accumulate(
    true: NP.NDArray[np.int32 | np.uint32 | np.uint64],
    pred: NP.NDArray[np.int32 | np.uint32 | np.uint64],
    num_cats: int,
    ignore_label: int,
    label_divisor: int,
    offset: int,
) -> PQStat:
    """
    Compares predicted segmentation with groundtruth, accumulates its
    metric.
    It is not assumed that instance ids are unique across different
    categories.
    See for example combine_semantic_and_instance_predictions.py in
    official PanopticAPI evaluation code for issues to consider when
    fusing category and instance labels.
    Instances ids of the ignored category have the meaning that id 0 is
    "void" and remaining ones are crowd instances.

    Parameters
    ----------
    label_true:
        A tensor that combines label array from categories and
        instances for ground truth.
    label_pred:
        A tensor that combines label array from categories and
        instances for the prediction.

    Returns
    -------
    The value of the metrics (iou, tp, fn, fp)
    """

    # Promote types
    true = true.astype(np.uint64)
    pred = pred.astype(np.uint64)

    num_cats_ = np.uint64(num_cats)
    ignore_label_ = np.uint64(ignore_label)
    label_divisor_ = np.uint64(label_divisor)
    offset_ = np.uint64(offset)
    zero_ = np.uint64(0)

    # Allocate results
    stat = PQStat(
        iou=np.zeros(num_cats_, dtype=np.float64),
        tp=np.zeros(num_cats_, dtype=np.float64),
        fn=np.zeros(num_cats_, dtype=np.float64),
        fp=np.zeros(num_cats_, dtype=np.float64),
    )

    # Pre-calculate areas for all groundtruth and predicted segments.
    true_areas = count_labels(true)
    pred_areas = count_labels(pred)

    # We assume the ignored segment has instance id = 0.
    true_ignored = ignore_label_ * label_divisor_ * offset_

    # Next, combine the groundtruth and predicted labels. Dividing up the
    # pixels based on which groundtruth segment and which predicted segment
    # they belong to, this will assign a different 64-bit integer label to
    # each choice of (groundtruth segment, predicted segment), encoded as
    #     gt_panoptic_label * offset + pred_panoptic_label.
    true_pred = true * offset_ + pred

    # For every combination of (groundtruth segment, predicted segment) with a
    # non-empty intersection, this counts the number of pixels in that
    # intersection.
    isec_areas = count_labels(true_pred)

    # Compute overall ignored overlap.
    # def prediction_ignored_overlap(pred_label):
    #     intersection_id = true_ignored + pred_label
    #     return intersection_areas.get(intersection_id, 0)

    # Sets that are populated with which segments groundtruth/predicted segments
    # have been matched with overlapping predicted/groundtruth segments
    # respectively.
    true_matched = set()
    pred_matched = set()

    # Calculate IoU per pair of intersecting segments of the same category.
    for intersection_id, intersection_area in isec_areas.items():
        true_label = intersection_id // offset_
        pred_label = intersection_id % offset_

        true_cat = true_label // label_divisor_
        pred_cat = pred_label // label_divisor_

        if true_cat != pred_cat:
            continue
        if pred_cat == ignore_label_:
            continue

        # Union between the groundtruth and predicted segments being compared
        # does not include the portion of the predicted segment that consists of
        # groundtruth "void" pixels.
        union = (
            true_areas[true_label]
            + pred_areas[pred_label]
            - intersection_area
            - isec_areas.get(true_ignored + pred_label, zero_)
        )
        iou = intersection_area / union
        if iou > 0.5:
            stat.tp[true_cat] += 1
            stat.iou[true_cat] += iou

            true_matched.add(true_label)
            pred_matched.add(pred_label)

    # Count false negatives for each category.
    for true_label in true_areas:
        if true_label in true_matched:
            continue
        true_cat = true_label // label_divisor_

        # Failing to detect a void segment is not a false negative.
        if true_cat == ignore_label_:
            continue

        try:
            stat.fn[true_cat] += 1
        except Exception:
            warn(f"True category {true_cat} is not valid! Treated as IGNORE!")

    # Count false positives for each category.
    for pred_label in pred_areas:
        if pred_label in pred_matched:
            continue
        # A false positive is not penalized if is mostly ignored in the
        # groundtruth.
        if (isec_areas.get(true_ignored + pred_label, zero_) / pred_areas[pred_label]) > 0.5:
            continue
        pred_cat = int(pred_label // label_divisor_)
        if pred_cat == ignore_label_:
            continue
        try:
            stat.fp[pred_cat] += 1
        except Exception:
            warn(f"Predicted category {pred_cat} is not valid! Treated as IGNORE!")

    return stat
