from abc import abstractmethod
from typing import Final, Generic, Sequence, TypeVar

from detectron2.evaluation import DatasetEvaluator as _D2_DatasetEvaluator
from typing_extensions import Self

__all__ = ["BaseEvaluator", "KEY_EVALUATE"]

Exposure = TypeVar("Exposure", bound=dict)
Outcome = TypeVar("Outcome", bound=dict)


KEY_EVALUATE: Final = "evaluate"


class BaseEvaluator(_D2_DatasetEvaluator, Generic[Exposure, Outcome]):
    @classmethod
    @abstractmethod
    def from_metadata(cls, dataset_name: str, **kwargs) -> Self:
        raise NotImplementedError

    @abstractmethod
    def process_item(self, exp: Exposure, out: Outcome) -> None:
        raise NotImplementedError

    def process(self, exps: Sequence[Exposure], outs: Sequence[Outcome]) -> None:
        assert len(exps) == len(outs)

        for exp, out in zip(exps, outs):
            if not self._has_truths(exp):
                continue
            self.process_item(exp, out)

    @staticmethod
    def _has_truths(exp: Exposure) -> bool:
        return exp.get(KEY_EVALUATE, True)
