"""
Panoptic Segmentation Evaluator
"""

from __future__ import annotations

import typing as T
import itertools
import dataclasses as D
import multiprocessing as mp
from tensordict import TensorDict, TensorDictBase
from functools import partial
from logging import warn

import numpy as np
import numpy.typing as NP
from tabulate import tabulate
import torch

from tqdm import tqdm
from evaluators.base import EvaluatorBase

from .prepare import get_panoptic_pair, put_panoptic_pair, put_panoptic_pair
from .utils import count_labels, stable_div

__all__ = ["PQEvaluator"]

class PQResult(T.NamedTuple):
    """
    Resulting PQ, RQ and SQ as a C-dimensional array, where C is the amount of
    classes.
    """

    PQ: NP.NDArray[np.float64]
    RQ: NP.NDArray[np.float64]
    SQ: NP.NDArray[np.float64]


class PQStat(T.NamedTuple):
    """
    Resulting PQ, RQ and SQ as a C-dimensional array, where C is the amount of
    classes.
    """

    iou: NP.NDArray[np.float64]
    tp: NP.NDArray[np.float64]
    fp: NP.NDArray[np.float64]
    fn: NP.NDArray[np.float64]

@D.dataclass(slots=True)
class PQEvaluator(EvaluatorBase):
    """
    Computes the Panoptic Quality (PQ) metric
    """

    thing_classes: list[int]
    thing_names: list[str]
    stuff_classes: list[int]
    stuff_names: list[str]
    label_divisor: int
    true_key: str
    pred_key: str
    ignore_label: int = -1
    offset: int = 2 ** 20
    metric_name: str = "PQ"
    
    def __post_init__(self):
        # Sanitize classes and names for stuff in cases where the predictor defines stuff as including (some) things
        if self.stuff_classes == self.thing_classes:
            self.stuff_classes = []
            self.stuff_names = []
        else:
            self.stuff_classes, self.stuff_names = map(
                list, zip(*[(label, name) for label, name in zip(self.stuff_classes, self.stuff_names) if label not in self.thing_classes])
            )

        # Sanity checks
        if self.ignore_label in self.thing_classes:
            raise ValueError(f"Ignore label {self.ignore_label} is a thing class!")
        if self.ignore_label in self.stuff_classes:
            raise ValueError(f"Ignore label {self.ignore_label} is a stuff class!")
        if len(self.thing_classes) != len(self.thing_names):
            raise ValueError(f"Thing classes and names have different lengths!")
        if len(self.stuff_classes) != len(self.stuff_names):
            raise ValueError(f"Stuff classes and names have different lengths!")
    
    def num_classes(self):
        return len(self.stuff_classes) + len(self.thing_classes)

    def _prepare(self, inputs: TensorDictBase, outputs: TensorDictBase, keys: T.Collection[str]) -> TensorDict | None:
        return put_panoptic_pair(inputs, outputs, keys, true_key = self.true_key, pred_key = self.pred_key)

    def _evaluate(self, results: TensorDict) -> dict[str, float]:
        true, pred = get_panoptic_pair(results)

        print("*** Evaluating PQ ***")

        pqs = []

        for true_chunk, pred_chunk in tqdm(zip(true, pred)):
            pq_chunk = accumulate(
                true_chunk.numpy(),
                pred_chunk.numpy(),
                num_cats=self.num_classes(),
                ignore_label=self.ignore_label,
                label_divisor=self.label_divisor,
                offset=self.offset,
            )
            pqs.append(pq_chunk)


        assert len(pqs) > 0
        assert len(pqs[0]) == 4

        pqs_total = PQStat(*map(partial(torch.sum, axis=0), zip(*pqs)))  # type: ignore
        res = evaluate_pq(
            iou=pqs_total.iou,
            tp=pqs_total.tp,
            fp=pqs_total.fp,
            fn=pqs_total.fn,
        )
        pqs_mean = PQStat(*map(partial(np.mean, axis=0), zip(*pqs)))  # type: ignore

        output = {}
        for key, values in {
            "pq": res.PQ,
            "sq": res.SQ,
            "rq": res.RQ,
        }.items():
            output[key] = values.mean()
            output[key + "_thing"] = values[self.thing_classes].mean()
            output[key + "_stuff"] = values[self.stuff_classes].mean()

        self.print_result(pqs_mean, res)

        return output

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
        print("PQ evaluation results:\n" + table)


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
    mask = (tp + fn + fp) == 0

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

