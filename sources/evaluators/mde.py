"""
Monocular Depth Estimation Evaluators
"""

from collections import OrderedDict, defaultdict
from functools import cached_property, reduce
from itertools import starmap
from typing import Final, Iterable, Optional

from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from tabulate import tabulate
from torch import Tensor
from typing_extensions import Self

from ._depth_utils import DepthAccumulator, DepthMetrics
from ._types import Exposures, Outcomes

logger = setup_logger(name=f"{__name__}_{comm.get_local_rank()}")

class MDEEvaluator(DatasetEvaluator):
    def __init__(
        self,
        *,
        ignored_label: int,
        label_divisor: int,
        thing_classes: Iterable[int],
        stuff_classes: Iterable[int],
        task_name="task_depth",
    ):

        self.task_name: Final = task_name
        self.ignored_label: Final = ignored_label
        self.label_divisor: Final = label_divisor
        self.thing_classes: Final = list(thing_classes)
        self.stuff_classes: Final = list(
            _id for _id in stuff_classes if _id not in self.thing_classes
        )
        self.metrics = DepthAccumulator()

    @classmethod
    def from_metadata(cls, dataset_name: str, **kwargs) -> Self:
        from detectron2.data import MetadataCatalog

        m = MetadataCatalog.get(dataset_name)

        thing_classes = list(m.thing_dataset_id_to_contiguous_id.values())
        stuff_classes = list(
            _id
            for _id in m.stuff_dataset_id_to_contiguous_id.values()
            if _id not in thing_classes
        )

        return cls(
            ignored_label=m.ignore_label,
            label_divisor=m.label_divisor,
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
            **kwargs,
        )

    @cached_property
    def num_classes(self):
        return len(self.stuff_classes) + len(self.thing_classes)

    def reset(self):
        self.metrics = DepthAccumulator()

    def process_item(
        self,
        input_: dict[str, Tensor],
        output: dict[str, Tensor],
    ):
        if not input_["has_truths"]:
            return

        depth_true = input_.get("depth")
        if depth_true is None or depth_true.max() == 0.0:
            return
        depth_true = depth_true.detach().cpu()

        depth_pred = output.get("depth")
        if depth_pred is None:
            raise ValueError("Output has no estimated depth map")
        depth_pred = depth_pred.detach().cpu()

        sem_seg = input_.get("sem_seg")
        if sem_seg is not None:
            valid_mask = (sem_seg != self.ignored_label).detach().cpu().numpy()
            sem_seg = sem_seg.detach().cpu().numpy()
        else:
            valid_mask = None

        self.metrics.update(
            depth_true=depth_true.numpy(),
            depth_pred=depth_pred.numpy(),
            valid_mask=valid_mask,
            category_mask=sem_seg,
        )

    def process(self, inputs: list[Exposures], outputs: list[Outcomes]):
        for input_, output in zip(inputs, outputs):
            self.process_item(input_, output)

    def evaluate(self) -> Optional[dict[str, dict[str, float]]]:
        comm.synchronize()
        self.metrics = reduce(
            lambda a, b: a and a.gather(b) or b, comm.gather(self.metrics), None
        )

        if not comm.is_main_process():
            return None

        logger.info("Combining depth evaluator results...")

        result = self.metrics.result()
        output = {}

        logger.info("Computing results for overall, things and stuff...")

        for field, value in result:
            output[field] = value

        for field, value in result[self.thing_classes]:
            output[field + "_th"] = value

        for field, value in result[self.stuff_classes]:
            output[field + "_st"] = value

        self.print_results(result)

        return {self.task_name: output}

    def print_results(self, dm: DepthMetrics):
        data = defaultdict(list)
        data[""].append("All")
        for field, value in dm:
            data[field].append(value)

        data[""].append("Things")
        for field, value in dm[self.thing_classes]:
            data[field].append(value)

        data[""].append("Stuff")
        for field, value in dm[self.stuff_classes]:
            data[field].append(value)

        table = tabulate(
            list(zip(*data.values())),
            headers=list(data.keys()),
            tablefmt="pipe",
            floatfmt=".2f",
            stralign="center",
            numalign="center",
        )
        logger.info("Depth evaluation results:\n" + table)
