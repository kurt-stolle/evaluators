import typing as T
import abc

from evaluators._protocol import CompatibleDict, TensorDict, TensorDictBase

MetricValueType = T.TypeVar("MetricValueType", bound=int | str | float | bool)
ResultType = T.TypeVar("ResultType", bound=CompatibleDict | None, covariant=True)

def _as_tensordict(x: CompatibleDict) -> TensorDictBase:
    if isinstance(x, TensorDictBase):
        return x
    if isinstance(x, dict):
        return TensorDict.from_dict(x)
    if hasattr(x, "to_tensordict"):
        return x.to_tensordict() # type: ignore
    raise TypeError(f"Expected a TensorDict, a Mapping or an object with a to_tensordict method, got {type(x)}")

class EvaluatorBase(T.Generic[ResultType, MetricValueType]):
    @abc.abstractmethod
    def _prepare(self, inputs: TensorDictBase, outputs: TensorDictBase, keys: frozenset[str]) -> ResultType:
        ...

    def prepare(self, inputs: CompatibleDict, outputs: CompatibleDict, keys: T.Collection[str]) -> TensorDictBase | None:
        inputs = _as_tensordict(inputs)
        outputs = _as_tensordict(outputs)
        keys = frozenset(keys)
        results = self._prepare(inputs, outputs, keys)
        return _as_tensordict(results) if results is not None else None

    @abc.abstractmethod
    def _evaluate(self, results: TensorDictBase, interms: dict[str, T.Any]) -> dict[str, MetricValueType]:
        ...

    def evaluate(self, results: CompatibleDict) -> dict[str, int | str | float | bool]:
        results = _as_tensordict(results)
        metrics = self._evaluate(results, T.cast(dict, interms))
        return metrics

# class EvaluatorBaseWithKeys(EvaluatorBase):
#     @property
#     @abc.abstractmethod
#     def _input_keys(self) -> T.Collection[str]:
#         ...

#     @property
#     @abc.abstractmethod
#     def _output_keys(self) -> T.Collection[str]:
#         ...

#     def _prepare(self, inputs: TensorDictBase, outputs: TensorDictBase, keys: frozenset[str]) -> ResultType:
#         # Check if all keys are already in results
#         if keys.issuperset({f"inputs.{k}" for k in self._input_keys}) and keys.issuperset({f"outputs.{k}" for k in self._output_keys}):
#             return None

#         res = {}

#         # Add input keys to result
#         for key in self._input_keys:
#             key_res = f"inputs.{key}"
#             if key_res not in keys:
#                 res[key_res] = inputs[key]

#         # Add output keys to result
#         for key in self._output_keys:
#             key_res = f"outputs.{key}"
#             if key_res not in keys:
#                 res[key_res] = outputs[key]

#         return TensorDict.from_dict(res)

    