from __future__ import annotations

import typing as T
from tensordict import TensorDictBase, TensorDict
import torch

TRUE_PANOPTIC = "true_panoptic"
PRED_PANOPTIC = "pred_panoptic"


def put_panoptic_pair(inputs: TensorDictBase, outputs: TensorDictBase, keys: T.Collection[str], *, true_key: str, pred_key: str) -> TensorDict | None:
    """
    Store the true and predicted panoptic segmentation masks for evaluation.
    """
    if TRUE_PANOPTIC in keys and PRED_PANOPTIC in keys:
        return None
    elif TRUE_PANOPTIC in keys or PRED_PANOPTIC in keys:
        raise ValueError(f"Both {TRUE_PANOPTIC} and {PRED_PANOPTIC} must be in keys or neither should be in keys.")

    # Check if sample has annotations
    true = inputs.get(true_key, None)
    if true is None:
        return None  # No annotations, skip sample
    pred = outputs.get(pred_key)

    # Masked predictions
    return TensorDict({
        "true_panoptic": true,
        "pred_panoptic": pred,
    })

def get_panoptic_pair(results: TensorDictBase) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Retrieve the true and predicted panoptic segmentation masks for evaluation.
    """
    true = results[TRUE_PANOPTIC]
    pred = results[PRED_PANOPTIC]
    return true, pred