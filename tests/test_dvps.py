import numpy as np
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from evaluators import dvps


def test_shared_data():
    sem = np.arange(3 * 6 * 6, dtype=np.int32).reshape(3, 6, 6)
    ins = np.arange(3 * 6 * 6, dtype=np.int32).reshape(3, 6, 6)
    dep = np.arange(3 * 6 * 6, dtype=np.float32).reshape(3, 6, 6)

    data = dvps._Data(
        semantic=dvps._DataTuple(sem, sem + 1),
        instance=dvps._DataTuple(ins, ins + 2),
        depth=dvps._DataTuple(dep, dep + 3),
    )

    with dvps._DataShared.Context(data) as data_shm:
        data_rec = data_shm.recover(slice(0, 3))

        for t, t_rec in zip(data, data_rec):
            for i, i_rec in zip(t, t_rec):
                assert (i == i_rec).all(), (i, i_rec)


def test_evaluator():
    stuff_classes = [0, 1, 2, 3]
    thing_classes = [4, 5, 6, 7]
    label_divisor = 1000

    height = 2
    width = 4
    num_frames = 10

    semantic_true = np.arange(height * width).reshape(height, width)
    semantic_pred = semantic_true.copy()
    semantic_pred[:, 2] = 1
    semantic_true[:, 1] = 255

    instance_true = np.arange(start=1, stop=4 * height * width + 1, step=4).reshape(height, width)
    instance_true[np.isin(instance_true, stuff_classes)] = 0
    instance_pred = instance_true.copy()
    instance_pred[:, 3] = 0

    depth_true = np.ones((height, width), dtype=np.float32)
    depth_pred = depth_true.copy()
    depth_pred[1, :] = 100

    evaluator = dvps.DVPSEvaluator(
        accumulators=[
            dvps.DVPQAccumulator(
                ignored_label=255,
                label_divisor=label_divisor,
                thing_classes=thing_classes,
                stuff_classes=stuff_classes,
                depth_threshold=0.25,
            ),
            dvps.DVPQAccumulator(
                ignored_label=255,
                label_divisor=label_divisor,
                thing_classes=thing_classes,
                stuff_classes=stuff_classes,
                depth_threshold=0.5,
            ),
            dvps.DVPQAccumulator(
                ignored_label=255,
                label_divisor=label_divisor,
                thing_classes=thing_classes,
                stuff_classes=stuff_classes,
                depth_threshold=0.1,
            ),
        ],
        frames={1, num_frames // 2},
    )

    for frame in range(num_frames):
        inputs = {
            "has_truths": True,
            "labels": torch.from_numpy(semantic_true * label_divisor + instance_true),
            "depth": torch.from_numpy(depth_true),
            "sequence_id": "sequence_00",
            "frame": frame,
        }
        outputs = {
            "panoptic_labels": (
                torch.from_numpy(semantic_true),
                torch.from_numpy(instance_true),
            ),
            "depth": torch.from_numpy(depth_pred),
        }

        evaluator.process([inputs], [outputs])

    result = evaluator.evaluate()

    assert result is not None
    assert "task_dvps" in result

    expected = {
        "PQ": 38.095,
        "PQ_th": 21.429,
        "PQ_st": 50.000,
        "RQ": 41.270,
        "RQ_th": 21.429,
        "RQ_st": 50.000,
        "SQ": 0.0,
        "SQ_th": 21.429,
        "SQ_st": 50.000,
    }

    for key, value in result["task_dvps"].items():
        assert value is not None

        expected_value = expected.pop(key)
        assert expected_value is not None

        # assert np.isclose(value, expected_value, atol=0.001), (
        #     value,
        #     expected_value,
        # )

        print(f"{key:10s}: {value:.3f}")

    assert len(expected) == 0, expected
