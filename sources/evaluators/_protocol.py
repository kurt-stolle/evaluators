"""
Implements the Evaluator protocol, which is used to compute metrics on the outputs of a model.
"""

from __future__ import annotations

import typing as T
from tensordict import TensorDict, TensorDictBase
import abc


import typing as T
import torch
from tensordict import TensorDict, TensorDictBase

__all__ = ["CompatibleDict", "Evaluator"]


@T.runtime_checkable
class _SupportsToTensordict(T.Protocol):
    def to_tensordict(self) -> TensorDict:
        ...


CompatibleDict: T.TypeAlias = TensorDictBase | _SupportsToTensordict | T.Mapping[str, torch.Tensor | TensorDictBase]


@T.runtime_checkable
class Evaluator(T.Protocol):
    """
    Protocol class for evaluators.
    Evaluators are stateless objects that are used to compute metrics on the outputs of a model.
    """

    @abc.abstractmethod
    def prepare(
        self, inputs: CompatibleDict, outputs: CompatibleDict, keys: T.Collection[str]
    ) -> TensorDictBase | None:
        """
        The prepare method should take the required inputs and outputs to compute the metrics and return a dictionary
        containing the prepared inputs and outputs, where the shapes the result must be consistent over all batches
        within a single dataset.

        This function is called on every distributed process.

        The keys should be SUFFICIENTLY UNIQUE to avoid collisions with other evaluators, except for the case where
        such a collision is intended, e.g. for computing the same metric on different subsets of the data.

        Parameters
        ----------
        inputs
            The inputs to the model, which potentially contain the targets.
        outputs
            The outputs of the model.
        keys
            The keys that are already present in the outputs dictionary, i.e. that have potentially been computed/added
            by another evaluator.
        """
        ...

    @abc.abstractmethod
    def evaluate(
        self, results: CompatibleDict
    ) -> dict[str, int | str | float | bool]:
        """
        Process the prepared inputs and outputs and return a dictionary containing the computed metrics. This function
        should not perform any distributed operations, e.g. all_gather, as this is handled by the trainer.

        The resulting dictionary must only contrain basic Python primitive types, e.g. int, float, str, bool.

        Parameters
        ----------
        results
            The prepared inputs and outputs.
        interms
            The intermediates that have been computed by other evaluators, which prevents us from having to recompute
            the same operation on the results multiple times.
        """
        ...
