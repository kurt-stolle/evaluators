from __future__ import annotations

from abc import abstractstaticmethod
from contextlib import contextmanager
from multiprocessing import shared_memory
from typing import Generic, Iterable, NamedTuple, TypeVar

import numpy as np
import numpy.typing as NP
from typing_extensions import Self

_MapType = np.float32 | np.int32
_ScalarType = TypeVar("_ScalarType", bound=np.generic, covariant=True)
_DataItem = TypeVar("_DataItem", bound=NamedTuple)


class EvalPair(NamedTuple):
    true: NP.NDArray[_MapType]
    pred: NP.NDArray[_MapType]


class SharedData(Generic[_DataItem]):
    """
    Move a data item to shared memory for multiprocess read-only access.
    """

    @abstractstaticmethod
    def create(*args, **kwargs) -> _DataItem:
        raise NotImplementedError("Create not implemented!")

    def __init__(self, data: _DataItem):
        # Store meta information for recovery
        self.shape = data[0][0].shape
        if not all(i.shape == self.shape for t in data for i in t):
            raise ValueError("Data entries are not all the same shape!")

        if not len(self.shape) == 3:
            raise ValueError(f"Shared data is restricted to concatenated items: {self.shape}")

        self.data_type = type(data)
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

    def __len__(self) -> int:
        return self.shape[0]

    @classmethod
    def from_list(cls, items: list[_DataItem]) -> Self:
        if len(items) == 0:
            raise NotImplementedError()

        num_records = len(items[0])

        d = tuple(cls._stack_pairs(item[n] for item in items) for n in range(num_records))

        return SharedData(cls.create(d))

    @staticmethod
    def _stack_pairs(values: Iterable[EvalPair]) -> EvalPair:
        true, pred = tuple(zip(*values))
        return EvalPair(true=np.stack(true, axis=0), pred=np.stack(pred, axis=0))

    def recover(self, data_slice: slice) -> _DataItem:
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

        data = tuple(EvalPair(arrays.pop(0), arrays.pop(0)) for _ in range(len(arrays) // 2))

        assert len(arrays) == 0, "Not all entries from mapped array were used!"

        return self.create(*data)

    def recover_all(self) -> _DataItem:
        return self.recover(slice(0, len(self)))

    def close(self) -> None:
        """
        Cleanup the shared memory object.
        """
        self.shm.close()
        self.shm.unlink()
        del self.shm

    @contextmanager
    @staticmethod
    def Context(data: _DataItem):
        """
        Move ``DVPSData`` to shared memory.

        Yields
        ------
            SharedData class
        """
        try:
            sd = SharedData(data)
            yield sd
        finally:
            sd.close()
