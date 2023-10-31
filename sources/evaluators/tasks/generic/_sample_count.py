"""
Implements a sample counter.
"""

import dataclasses as D
import typing as T
import torch
from evaluators.base import EvaluatorBase, TensorDictBase
from tensordict import TensorDictBase, TensorDict

__all__ = ["SampleCountEvaluator"]

@D.dataclass(slots=True)
class SampleCountEvaluator(EvaluatorBase):
    """Evaluator that counts the number of samples in a dataset."""

    def _prepare(self, inputs: TensorDictBase, outputs: TensorDictBase, keys: T.Sequence[str]) -> TensorDict | None:
        if self._key in keys:
            return
        return TensorDict.from_dict({self._key: torch.tensor([inputs.batch_size[0]])})

    def _evaluate(self, results: TensorDictBase) -> dict[str, int]:
        total_count = results[self._key].sum().item()
        return {"sample_amount": total_count}
