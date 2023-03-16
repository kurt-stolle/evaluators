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
