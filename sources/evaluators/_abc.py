from abc import ABC, abstractmethod
from typing import Generic, Iterable, NamedTuple, TypeVar

import numpy as np
import numpy.typing as NP
from typing_extensions import Self

__all__ = ["MetricResult", "MetricAccumulator"]

MetricItem = TypeVar("MetricItem", bound=NamedTuple)
MetricState = TypeVar("MetricState", bound=NamedTuple)


class MetricResult(ABC):
    fields: list[str] = []

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, categories: NP.ArrayLike) -> Self:
        pass

    def __iter__(self) -> Iterable[tuple[str, NP.NDArray[np.float64]]]:
        for field in self.fields:
            yield field, getattr(self, field)


class MetricAccumulator(ABC, Generic[MetricItem, MetricState]):
    def update(self, item: MetricItem) -> MetricState:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def evaluate(self, updates: list[MetricState]) -> dict[str, float]:
        raise NotImplementedError


Exposure = TypeVar("Exposure", bound=dict)
Outcome = TypeVar("Outcome", bound=dict)


class Evaluator(ABC, Generic[Exposure, Outcome]):
    def __init__(self, task_name: str):
        self.task_name = task_name
        self._items = []

    def reset(self) -> None:
        self._items = []

    @classmethod
    @abstractmethod
    def from_metadata(cls, dataset_names: str) -> Self:
        pass

    @abstractmethod
    def process(self, exposure: Exposure, outcome: Outcome) -> None:
        pass

    @abstractmethod
    def evaluate(self) -> dict[str, dict[str, float]]:
        pass
