"""
This package implements the Detectron2 interface for the STQ metric.

The STQ metric is intentionally implemented in a way that is independent
of a specific machine learning framework.
"""
import multiprocessing as mp
from typing import (
    Any, Iterable, Iterator, MutableMapping, NamedTuple, Optional, Sequence,
    TypedDict, TypeVar,
)

import numpy as np
import numpy.typing as NP
from detectron2.data import MetadataCatalog
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from torch import Tensor

from ._shm import EvalPair, SharedData
from ._types import Exposures, Outcomes
from .base_evaluator import BaseEvaluator

__all__ = ["STEPEvaluator"]

_logger = setup_logger(name=f"{__name__}")

_LabelType = np.int64


def compute_iou(
    confusion: NP.NDArray[np.float64],
) -> NP.NDArray[np.float64]:
    tps = confusion.diagonal()
    fps = confusion.sum(axis=0) - tps
    fns = confusion.sum(axis=1) - tps
    unions = tps + fps + fns
    ious = tps / np.maximum(unions, np.finfo(np.float32).eps)

    mask = np.where(unions > 0, False, True)

    return np.ma.MaskedArray(ious, mask=mask)


class STQData(NamedTuple):
    labels: EvalPair


class STQItem(NamedTuple):
    sequence_id: str
    frame: int
    data: STQData


class STQShared(SharedData):
    @staticmethod
    def create(*args, **kwargs) -> STQData:
        return STQData(*args, **kwargs)


class STQResult(NamedTuple):
    stq: np.float64
    stq_per_seq: NP.NDArray[np.float64]
    aq: np.float64
    aq_per_seq: NP.NDArray[np.float64]
    iou: np.float64
    iou_per_seq: NP.NDArray[np.float64]
    length_per_seq: NP.NDArray[np.int64]


class STQResultSingle(NamedTuple):
    stq: np.float64
    aq: np.float64
    aq_sum: np.float64
    aq_amount: int
    iou: np.float64
    iou_confusion: NP.NDArray[np.float64]
    length: int


class STQAccumulator(mp.Process):
    """
    Segmentation and Tracking Quality (STQ) metric.
    """

    _iou_confusion: Optional[NP.NDArray[np.float64]]
    _pred: MutableMapping[_LabelType, int]
    _true: MutableMapping[_LabelType, int]
    _isec: MutableMapping[_LabelType, int]
    _seq_len: int

    def __init__(
        self,
        *,
        num_classes: int,
        things_list: Sequence[int],
        ignored_label: int,
        max_instances_per_category: int,
        offset: int,
        sequence_id: str,
        data: STQShared,
        queue: mp.Queue,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.ignored_label = ignored_label
        self.things_list = things_list
        self.label_divisor = max_instances_per_category
        self.offset = np.int64(offset)
        self.queue = queue
        self.data = data
        self.sequence_id = sequence_id

        if ignored_label >= num_classes:
            self._confusion_matrix_size = num_classes + 1
            self._include_indices = np.arange(self.num_classes)
        else:
            self._confusion_matrix_size = num_classes
            self._include_indices = np.array([i for i in range(num_classes) if i != self.ignored_label])

        lower_bound = num_classes * max_instances_per_category
        if offset < lower_bound:
            raise ValueError(
                "The provided offset %d is too small. No guarantess "
                "about the correctness of the results can be made. "
                "Please choose an offset that is higher than num_classes"
                " * max_instances_per_category = %d" % (offset, lower_bound)
            )

        self.reset()

    def reset(self):
        """
        Reset all states that accumulated data.
        """
        self._iou_confusion = None
        self._pred = {}
        self._true = {}
        self._isec = {}
        self._seq_len = 0

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ):
        """
        Accumulates the segmentation and tracking quality statistics.

        Panoptic label maps are defined as:
            :code:`semantic_map * max_instances_per_category + instance_map`

        Parameters
        ----------
        y_true:
            The ground-truth panoptic label map for a video frame.
        y_pred:
            The predicted panoptic label map for a video frame.
        """
        y_true = y_true.astype(_LabelType)
        y_pred = y_pred.astype(_LabelType)

        semantic_label = y_true // self.label_divisor
        semantic_prediction = y_pred // self.label_divisor

        # Check if the ignore value is outside the range [0, num_classes]. If yes,
        # map `_ignore_label` to `_num_classes`, so it can be used to create the
        # confusion matrix.
        if self.ignored_label > self.num_classes:
            semantic_label = np.where(
                np.not_equal(semantic_label, self.ignored_label),
                semantic_label,
                self.num_classes,
            )
            semantic_prediction = np.where(
                np.not_equal(semantic_prediction, self.ignored_label),
                semantic_prediction,
                self.num_classes,
            )

        self._seq_len += 1

        # Compute confusion matrix
        iou_confusion = confusion_matrix(
            np.reshape(semantic_label, [-1]),
            np.reshape(semantic_prediction, [-1]),
            labels=list(range(self._confusion_matrix_size)),
        )

        if self._iou_confusion is None:
            self._iou_confusion = iou_confusion
        else:
            self._iou_confusion += iou_confusion

        # Instance label
        instance_label = y_true % self.label_divisor

        label_mask = np.zeros_like(semantic_label, dtype=bool)
        prediction_mask = np.zeros_like(semantic_prediction, dtype=bool)

        for things_class_id in self.things_list:
            label_mask = np.logical_or(label_mask, np.equal(semantic_label, things_class_id))
            prediction_mask = np.logical_or(prediction_mask, np.equal(semantic_prediction, things_class_id))

        # Select the `crowd` region of the current class. This region is
        # encoded instance id `0`.
        is_crowd = np.logical_and(np.equal(instance_label, 0), label_mask)

        # Select the non-crowd region of the corresponding class as the `crowd`
        # is ignored for the tracking term.
        label_mask = np.logical_and(label_mask, np.logical_not(is_crowd))

        # Do not punish id assignment for regions that are annotated as `crowd`
        # in the ground-truth.
        prediction_mask = np.logical_and(prediction_mask, np.logical_not(is_crowd))

        # Compute and update areas of ground-truth, predictions and
        # intersections.
        self._update_dict_stats(self._pred, y_pred[prediction_mask])
        self._update_dict_stats(self._true, y_true[label_mask])

        isec_mask = np.logical_and(label_mask, prediction_mask)
        isec_lbls = y_true[isec_mask] * self.offset + y_pred[isec_mask]

        self._update_dict_stats(self._isec, isec_lbls)

    @staticmethod
    def _update_dict_stats(
        stat_dict: MutableMapping[_LabelType, int],
        id_array: NP.NDArray[_LabelType],
    ):
        ids, counts = np.unique(id_array, return_counts=True)
        for idx, count in zip(ids, counts):
            if idx in stat_dict:
                stat_dict[idx] += count
            else:
                stat_dict[idx] = count

    def result(self) -> STQResultSingle:
        """
        Compute STQ and related metrics.

        Returns
        -------
        Results object.
        """
        assert self._iou_confusion is not None
        assert self._seq_len > 0

        # Compute association quality (AQ)
        aq_amount = len(self._true)
        aq_sum = np.float64(0.0)

        for true_lbl, true_size in self._true.items():
            inner_sum = 0.0
            for pred_lbl, pred_size in self._pred.items():
                tpa_key = self.offset * true_lbl + pred_lbl
                tpa = self._isec.get(tpa_key)
                if tpa is None:
                    continue

                fpa = pred_size - tpa
                fna = true_size - tpa

                inner_sum += tpa * (tpa / (tpa + fpa + fna))

            aq_sum += 1.0 / true_size * inner_sum

        aq = (aq_sum / aq_amount) if aq_amount > 0 else np.float64(0.0)

        # Compute IoU scores.
        # The rows correspond to ground-truth and the columns to predictions.
        # Remove fp from confusion matrix for the void/ignore class.
        removal_matrix = np.zeros_like(self._iou_confusion)
        removal_matrix[self._include_indices, :] = 1.0

        confusion = self._iou_confusion * removal_matrix

        iou = compute_iou(confusion).mean()
        stq = np.sqrt(aq * iou)

        return STQResultSingle(
            stq=stq,
            aq=aq,
            aq_sum=aq_sum,
            aq_amount=aq_amount,
            iou=iou,
            iou_confusion=confusion,
            length=self._seq_len,
        )

    @staticmethod
    def accumulate(seq_data: Sequence[STQResultSingle]) -> STQResult:
        assert len(seq_data) > 0

        (
            stq_per_seq,
            aq_per_seq,
            aq_sum_per_seq,
            aq_amount_per_seq,
            iou_per_seq,
            iou_confusion_per_seq,
            length_per_seq,
        ) = tuple(zip(*seq_data))

        aq_per_seq = np.asarray(aq_per_seq)
        aq_sum_per_seq = np.asarray(aq_sum_per_seq)
        aq_amount_per_seq = np.asarray(aq_amount_per_seq)
        aq_mean = np.sum(aq_sum_per_seq) / np.maximum(np.sum(aq_amount_per_seq), 1e-12)

        # total_confusion = np.sum(iou_confusion_per_seq, axis=0, keepdims=True)
        total_confusion = np.zeros_like(iou_confusion_per_seq[0])
        for iou_confusion in iou_confusion_per_seq:
            total_confusion += iou_confusion

        iou_per_seq = np.asarray(iou_per_seq)
        iou_mean = compute_iou(total_confusion).mean()

        stq_per_seq = np.asarray(stq_per_seq)
        stq_mean = np.sqrt(aq_mean * iou_mean)

        return STQResult(
            stq=stq_mean,
            stq_per_seq=stq_per_seq,
            aq=aq_mean,
            aq_per_seq=aq_per_seq,
            iou=iou_mean,
            iou_per_seq=iou_per_seq,
            length_per_seq=length_per_seq,
        )

    def run(self):
        data: STQData = self.data.recover_all()

        for y_true, y_pred in zip(data.labels.true, data.labels.pred):
            self.update(y_true=y_true, y_pred=y_pred)
        res = self.result()
        self.queue.put((self.sequence_id, res))


class STEPExposure(TypedDict):
    sequence_id: str | int
    frame: int
    evaluate: bool
    labels: Tensor


class STEPOutcome(TypedDict):
    panoptic_seg: tuple[Tensor, None]


class STEPEvaluator(BaseEvaluator):
    task_name = "task_step"

    def __init__(
        self,
        *,
        thing_classes: Sequence[int],
        stuff_classes: Sequence[int],
        ignored_label: int,
        label_divisor: int,
    ):
        # Properties from metadata
        self.stuff_classes = list(stuff_classes)
        self.thing_classes = list(thing_classes)
        self.ignored_label = ignored_label
        self.label_divisor = label_divisor

        assert len(self.stuff_classes) > 0
        assert self.ignored_label >= 0

        # State
        self.last_frame: dict[Any, Any] = {}
        self._items: list[STQItem] = []

    @classmethod
    def from_metadata(cls, dataset_names: str | Sequence[str], **kwargs):
        from ._meta import read_metadata
        m = read_metadata(dataset_names)

        thing_classes = list(m["thing_translations"].values())
        stuff_classes = [id_ for id_ in m["stuff_translations"].values() if id_ not in thing_classes]

        return cls(
            ignored_label=m["ignore_label"],
            label_divisor=m["label_divisor"],
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
            **kwargs,
        )

    def reset(self):
        self._items = []
        self._last_frame = {}

    def process_item(self, input_: dict[str, Any], output: dict[str, Any]) -> None:
        true = input_["labels"]

        # Sanity check: do frames in a sequence appear sequentially?
        sequence_id = str(input_["sequence_id"])
        frame = input_["frame"]
        frame_previous = self.last_frame.get(sequence_id)

        assert frame_previous is None or frame_previous < frame, (
            f"Frame {frame} not processed in the right order for "
            f"sequence {sequence_id}. Last frame was: {frame_previous}"
        )

        self.last_frame[sequence_id] = frame

        pred, _ = output["panoptic_seg"]  # type: ignore
        pred[pred == -1] = self.ignored_label * self.label_divisor
        pred = pred.cpu().numpy()

        # Update metric
        self._items.append(
            STQItem(
                sequence_id,
                frame,
                STQData(EvalPair(true=true.numpy(), pred=pred)),
            )
        )

    def evaluate(self):
        comm.synchronize()

        _logger.info("Moving data to shared memory...")
        seq_shm_split = _split_per_seq(self._items)
        assert len(self._items) == 0, "Not all items were consumed"

        seq_shm_all = comm.gather(seq_shm_split)

        # self._items = comm.gather(self._items)  # type: ignore
        # self._items = list(itertools.chain(*self._items))  # type: ignore
        if not comm.is_main_process():
            assert seq_shm_all is None or len(seq_shm_all) == 0
            return
        assert len(seq_shm_all) > 0

        # Merge shared memory sequence dicts for each GPU
        seq_shm = {}
        for item in seq_shm_all:
            assert item is not None
            assert all(key not in seq_shm for key in item.keys())

            for key in list(item.keys()):
                seq_shm[key] = item.pop(key)

            assert len(item) == 0

        queue = mp.Queue()
        processes: list[STQAccumulator] = []
        try:
            _logger.info("Starting concurrent sequence evaluation...")
            for seq_id, seq_data in seq_shm.items():
                p = STQAccumulator(
                    num_classes=len(self.thing_classes) + len(self.stuff_classes),
                    things_list=self.thing_classes,
                    ignored_label=self.ignored_label,
                    max_instances_per_category=self.label_divisor,
                    offset=2**20,
                    sequence_id=seq_id,
                    queue=queue,
                    data=seq_data,
                )
                p.start()
                processes.append(p)

            results = {}
            while len(seq_shm) > 0:
                seq_id, seq_res = queue.get()
                results[seq_id] = seq_res
                seq_shm.pop(seq_id).close()

            for p in processes:
                p.join()
        finally:
            if seq_shm is not None:
                for shm in seq_shm.values():
                    shm.close()

        _logger.info(f"Computing overall results")
        result_all = STQAccumulator.accumulate(list(results.values()))

        result = {
            "STQ": 100.0 * float(result_all.stq),
            "AQ": 100.0 * float(result_all.aq),
            "IoU": 100.0 * float(result_all.iou),
        }

        self.print_result(result_all, ids=list(results.keys()))

        return {self.task_name: result}

    def print_result(self, stq: STQResult, *, ids: list[str]):
        data = []
        for i, id in enumerate(ids):
            data.append(
                [
                    id,
                    stq.stq_per_seq[i] * 100,
                    stq.aq_per_seq[i] * 100,
                    stq.iou_per_seq[i] * 100,
                    str(stq.length_per_seq[i]),
                ]
            )
        data.append(
            [
                "All",
                stq.stq * 100,
                stq.aq * 100,
                stq.iou * 100,
                str(sum(l for l in stq.length_per_seq)),
            ]
        )
        table = tabulate(
            data,
            headers=["", "STQ", "AQ", "IoU", "#"],
            tablefmt="pipe",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )
        _logger.info("STEP evaluation results:\n" + table)


def _stack_tuples(values: Iterable[EvalPair]) -> EvalPair:
    """
    Stack a list of ``DVPSTuple`` to a single ``DVPSTuple``.
    """
    true, pred = tuple(zip(*values))
    return EvalPair(true=np.stack(true, axis=0), pred=np.stack(pred, axis=0))


def _split_per_seq(
    items: list[STQItem],
) -> dict[str, STQShared]:
    """
    Create a mapping of sequence IDs to a concatenated data structure.
    """

    # Create mapping ID -> Items
    seq_items: dict[str, list[STQItem]] = {}
    for _ in range(len(items)):
        i = items.pop(0)
        seq_id = i.sequence_id
        if seq_id not in seq_items:
            seq_items[seq_id] = []
        seq_items[seq_id].append(i)

    # Sort by frame number and concatenate to get mapping ID -> Data
    seq_data: dict[str, STQShared] = {}
    for k, v in seq_items.items():
        v = sorted(v, key=lambda i: i.frame)
        d = STQData(
            labels=_stack_tuples(i.data.labels for i in v),
        )

        seq_data[k] = STQShared(d)

        del d
        del v

    return seq_data
