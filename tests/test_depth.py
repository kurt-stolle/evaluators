import numpy as np

import evaluators.depth as depth_module


def test_compute():
    a = np.random.rand(3, 3)
    b = np.random.rand(3, 3)

    res = depth_module.compute_metrics(a, b)

    assert res is not None
