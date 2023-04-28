"""
Depth-Aware Video Panoptic Segmentation Evaluators
"""

from __future__ import annotations

import multiprocessing as mp
from contextlib import contextmanager
from functools import cached_property, partial
from multiprocessing import shared_memory
from statistics import mean
from typing import Iterable, NamedTuple, Sequence, Set, TypeVar

import numpy as np
import numpy.typing as NP
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger

from ._utils import stable_div
from .panseg import accumulate

__all__ = ["DVPSEvaluator"]

_MapType = np.float32 | np.int32
_ScalarType = TypeVar("_ScalarType", bound=np.generic, covariant=True)

_logger = setup_logger(name=__name__)


class _DataTuple(NamedTuple):
    true: NP.NDArray[_MapType]
    pred: NP.NDArray[_MapType]


class _Data(NamedTuple):
    semantic: _DataTuple
    instance: _DataTuple
    depth: _DataTuple


class _DataShared:
    """
    Move a ``DVPSData`` item to shared memory for multiprocess read-only access.
    """

    def __init__(self, data: _Data):
        # Store meta information for recovery
        self.shape = data[0][0].shape
        if not all(i.shape == self.shape for t in data for i in t):
            raise ValueError("Data entries are not all the same shape!")

        if not len(self.shape) == 3:
            raise ValueError(f"Shared data is restricted to concatenated items: {self.shape}")

        self.dtypes = [i.dtype for t in data for i in t]
        self.nbytes = [i.nbytes for t in data for i in t]

        # Allocate shared memory
        self.shm = shared_memory.SharedMemory(create=True, size=sum(self.nbytes))

        # Move data into shared memory
        offset = 0
        for i, (dtype, nbytes) in enumerate(zip(self.dtypes, self.nbytes)):
            item_mem = data[i // 2][i % 2]
            item_shm = np.ndarray(
                shape=self.shape,
                dtype=dtype,
                buffer=self.shm.buf[offset : offset + nbytes],  # noqa: E203
            )
            item_shm[:] = item_mem[:]

            offset += nbytes

    def recover(self, data_slice: slice) -> _Data:
        """
        Recover a ``DVPSData`` class from shared memory.
        """
        arrays = []
        offset = 0
        for dtype, nbytes in zip(self.dtypes, self.nbytes):
            rec = np.ndarray(
                shape=self.shape,
                dtype=dtype,
                buffer=self.shm.buf[offset : offset + nbytes],  # noqa: E203
            )

            assert rec.ndim == 3, rec.ndim
            assert rec.shape[0] >= data_slice.stop

            arrays.append(rec[data_slice, :, :])

            offset += nbytes

        assert len(arrays) % 2 == 0, len(arrays)

        data = _Data(
            _DataTuple(arrays.pop(0), arrays.pop(0)),
            _DataTuple(arrays.pop(0), arrays.pop(0)),
            _DataTuple(arrays.pop(0), arrays.pop(0)),
        )

        assert len(arrays) == 0, "Not all entries from mapped array were used!"

        return data

    def close(self) -> None:
        """
        Cleanup the shared memory object.
        """
        self.shm.close()
        self.shm.unlink()
        del self.shm

    @contextmanager
    @staticmethod
    def Context(data: _Data):
        """
        Move ``DVPSData`` to shared memory.

        Yields
        ------
            SharedData class
        """
        try:
            sd = _DataShared(data)
            yield sd
        finally:
            sd.close()


class _Item(NamedTuple):
    sequence_id: str
    frame: int
    data: _Data


class _VPQ(NamedTuple):
    iou_per_class: NP.NDArray[np.float64]
    tp_per_class: NP.NDArray[np.float64]
    fn_per_class: NP.NDArray[np.float64]
    fp_per_class: NP.NDArray[np.float64]


class DVPQAccumulator:
    """
    Stateless class for accumulating the DVPQ metric. The metric is computed by
    taking the VPQ of pixels at which the depth prediction has an absolute
    relative error that is less than a configured threshold value.
    """

    def __init__(
        self,
        *,
        ignored_label: int,
        label_divisor: int,
        thing_classes: list[int],
        stuff_classes: list[int],
        depth_threshold: float,
        offset: int = 2**20,
    ):
        self.ignored_label = ignored_label

        self.label_divisor = label_divisor
        self.thing_classes = thing_classes
        self.stuff_classes = stuff_classes

        self.offset = offset

        # See ViP-DeepLab paper sec. 3.3
        self.depth_threshold = depth_threshold

        lower_bound = self.num_categories * self.label_divisor
        if offset < lower_bound:
            raise ValueError(
                "The provided offset %d is too small. No guarantess "
                "about the correctness of the results can be made. "
                "Please choose an offset that is higher than num_classes"
                " * max_instances_per_category = %d" % (offset, lower_bound)
            )

    @cached_property
    def num_categories(self) -> int:
        return len(self.thing_classes) + len(self.stuff_classes)

    @classmethod
    def from_metadata(cls, dataset_name: str, **kwargs) -> DVPQAccumulator:
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

    def update(self, data_shared: _DataShared, data_slice: slice) -> _VPQ:
        # Read data from shared memory
        data = data_shared.recover(data_slice)

        all_classes = self.thing_classes + self.stuff_classes + [self.ignored_label]
        assert np.isin(data.semantic.true, all_classes).all(), np.unique(data.semantic.true)
        assert np.isin(data.semantic.pred, all_classes).all(), np.unique(data.semantic.pred)

        # Promote integer type s.t. we can multiply with offset and label
        # divisor
        semantic_pred = data.semantic.pred.astype(np.int32)
        semantic_true = data.semantic.true.astype(np.int32)

        # Create label matrices
        labels_pred = _cat_batch(semantic_pred * self.label_divisor + data.instance.pred)
        labels_true = _cat_batch(semantic_true * self.label_divisor + data.instance.true)

        # Ignore predictions when the absolute relative error (ARE) is greater
        # than the threshold value
        abs_rel = 0.0
        if self.depth_threshold > 0.0:
            depth_pred = _cat_batch(data.depth.pred)
            depth_true = _cat_batch(data.depth.true)
            depth_mask = depth_true > 0

            if not depth_true[depth_mask].any():
                raise ValueError("Annotated depth mask has no valid entries")

            depth_pred_valid = depth_pred[depth_mask]
            depth_true_valid = depth_true[depth_mask]

            abs_rel = stable_div(np.abs(depth_pred_valid - depth_true_valid), depth_true_valid)

            pred_in_mask = labels_pred[:, : depth_pred.shape[1]]
            pred_in_depth_mask = pred_in_mask[depth_mask]

            ignored_pred_mask = abs_rel > self.depth_threshold

            pred_in_depth_mask[ignored_pred_mask] = self.ignored_label * self.label_divisor
            pred_in_mask[depth_mask] = pred_in_depth_mask
            labels_pred[:, : depth_pred.shape[1]] = pred_in_mask

        return _VPQ(
            *accumulate(
                labels_true,  # type: ignore
                labels_pred,  # type: ignore
                self.num_categories,
                self.ignored_label,
                self.label_divisor,
                self.offset,
            )
        )

    def reduce(self, results: list[_VPQ]) -> dict[str, float]:
        total_iou = np.stack([result[0] for result in results]).sum(axis=0)
        total_tp = np.stack([result[1] for result in results]).sum(axis=0)
        total_fn = np.stack([result[2] for result in results]).sum(axis=0)
        total_fp = np.stack([result[3] for result in results]).sum(axis=0)

        sq = stable_div(total_iou, total_tp)
        rq = stable_div(
            total_tp,
            total_tp + 0.5 * total_fn + 0.5 * total_fp,
        )
        pq = sq * rq

        # Mask out metrics that have a sum total of 0 for TP, FN and FP
        mask = np.equal(total_tp + total_fn + total_fp, 0)

        # Use MaskedArray to create masked variants
        PQ = np.ma.MaskedArray(pq, mask)
        SQ = np.ma.MaskedArray(sq, mask)
        RQ = np.ma.MaskedArray(rq, mask)

        # Compile result for PQ, SQ and RQ and thing/stuff variants
        result = {
            "PQ": float(PQ.mean()),
            "PQ_th": float(PQ[self.thing_classes].mean()),
            "PQ_st": float(PQ[self.stuff_classes].mean()),
            "SQ": float(SQ.mean()),
            "SQ_th": float(SQ[self.thing_classes].mean()),
            "SQ_st": float(SQ[self.stuff_classes].mean()),
            "RQ": float(RQ.mean()),
            "RQ_th": float(RQ[self.thing_classes].mean()),
            "RQ_st": float(RQ[self.stuff_classes].mean()),
        }

        return result


class DVPSEvaluator(DatasetEvaluator):
    """
    Implements metrics for the Depth-Aware Video Panoptic Segmentation (DVPS)
    task.

    See: https://github.com/joe-siyuan-qiao/ViP-DeepLab
    """

    task_name = "task_dvps"

    def __init__(
        self,
        accumulators: list[DVPQAccumulator],
        frames: Iterable[int] | Set[int],
    ):
        if len(accumulators) == 0:
            raise ValueError(f"No accumulators were provided.")

        super().__init__()

        self._accumulators = accumulators
        self._items: list[_Item] = []
        self._frames = set(frames)

        self.reset()

    @classmethod
    def from_metadata(
        cls, dataset_name: str, thresholds: Sequence[float], frames: Sequence[int] | Set[int], **kwargs
    ) -> DVPSEvaluator:
        get_accumulator = partial(DVPQAccumulator.from_metadata, dataset_name)
        return cls(accumulators=[get_accumulator(threshold=t, **kwargs) for t in thresholds], frames=frames)

    def reset(self):
        self._items = []

    def _read_item(self, x, y) -> None:
        true_labels = x["labels"].numpy()
        true_semantic = (true_labels // 1000).astype(np.uint16)
        true_instance = (true_labels % 1000).astype(np.uint16)

        pred_semantic, pred_instance = y["panoptic_labels"]
        pred_semantic = pred_semantic.detach().cpu().numpy().astype(np.uint16)
        pred_instance = pred_instance.detach().cpu().numpy().astype(np.uint16)

        true_depth = x["depth"].detach()
        if not (true_depth > 0).any():
            return
            # _logger.warn(
            #     "No valid depth annotations for sample: "
            #     + x.get("image_id", "?")
            # )

        self._items.append(
            _Item(
                sequence_id=x["sequence_id"],
                frame=x["frame"],
                data=_Data(
                    instance=_DataTuple(true=true_instance, pred=pred_instance),  # type: ignore
                    semantic=_DataTuple(true=true_semantic, pred=pred_semantic),  # type: ignore
                    depth=_DataTuple(
                        true=true_depth.cpu().numpy().astype(np.float32),
                        pred=y["depth"].detach().cpu().numpy().astype(np.float32),
                    ),
                ),
            )
        )

    def process(self, inputs, outputs):
        for x, y in zip(inputs, outputs):
            if not x["evaluate"]:
                return
            if x.get("labels") is None:
                continue
            self._read_item(x, y)

    def evaluate(self):
        """
        Synchronize and evaluate score.
        """

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

        # Result is a key-value dictionary
        result: dict[str, list[float]] = {}

        # Merge shared memory sequence dicts for each GPU
        seq_shm = {}
        for item in seq_shm_all:
            assert item is not None
            assert all(key not in seq_shm for key in item.keys())

            for key in list(item.keys()):
                seq_shm[key] = item.pop(key)

            assert len(item) == 0

        try:
            _logger.info("Starting concurrent sequence evaluation...")

            with mp.Pool() as p:
                for frames in self._frames:
                    _logger.info(f"Evaluating DVPS metrics using window size: {frames}")

                    seq_slices = _split_slices(seq_shm, frames)

                    shdata = []
                    slices = []
                    for seq_id, slices in seq_slices.items():
                        slices += slices
                        shdata += [seq_shm[seq_id]] * len(slices)

                    acc_results = {
                        acc: p.starmap_async(acc.update, zip(shdata, slices))
                        # acc: itertools.starmap(acc.update, zip(shdata, slices))
                        for acc in self._accumulators
                    }
                    for acc, upd_async in acc_results.items():
                        upd = upd_async.get()
                        upd = list(upd)
                        red = acc.reduce(upd)

                        for key, value in red.items():
                            result[key] = result.get(key, []) + [value]
            p.join()
        finally:
            if seq_shm is not None:
                for shm in seq_shm.values():
                    shm.close()

        return {self.task_name: {key: float(mean(values)) * 100.0 for key, values in result.items()}}


def _cat_batch(a: NP.NDArray[_ScalarType], axis=1) -> NP.NDArray[_ScalarType]:
    return np.concatenate([*a], axis=axis)


def _stack_tuples(values: Iterable[_DataTuple]) -> _DataTuple:
    """
    Stack a list of ``DVPSTuple`` to a single ``DVPSTuple``.
    """
    true, pred = tuple(zip(*values))
    return _DataTuple(true=np.stack(true, axis=0), pred=np.stack(pred, axis=0))


def _split_per_seq(
    items: list[_Item],
) -> dict[str, _DataShared]:
    """
    Create a mapping of sequence IDs to a concatenated data structure.
    """

    # Create mapping ID -> Items
    seq_items: dict[str, list[_Item]] = {}
    for _ in range(len(items)):
        i = items.pop(0)
        seq_id = i.sequence_id
        if seq_id not in seq_items:
            seq_items[seq_id] = []
        seq_items[seq_id].append(i)

    # Sort by frame number and concatenate to get mapping ID -> Data
    seq_data: dict[str, _DataShared] = {}
    for k, v in seq_items.items():
        v = sorted(v, key=lambda i: i.frame)
        d = _Data(
            semantic=_stack_tuples(i.data.semantic for i in v),
            instance=_stack_tuples(i.data.instance for i in v),
            depth=_stack_tuples(i.data.depth for i in v),
        )

        seq_data[k] = _DataShared(d)

        del d
        del v

    return seq_data


def _split_slices(seq_data: dict[str, _DataShared], num_frames: int) -> dict[str, list[slice]]:
    """
    Generate indices that cover `num_frames` annotated images of a sequence.

    Parameters
    ----------
    sequence_ids
        List of sequence IDs
    num_frames
        Amount of frames to take from each sequence

    Yields
    ------
        List of indices to evaluate samples at
    """

    seq_slice: dict[str, list[slice]] = {}

    for seq_id, data in seq_data.items():
        seq_len = data.shape[0]
        seq_slice[seq_id] = [slice(frame, frame + num_frames) for frame in range(seq_len - num_frames + 1)]

    return seq_slice
