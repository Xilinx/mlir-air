import numpy as np
from comparators import regression_gate


def test_regression_gate_pass_and_fail():
    a = np.random.default_rng(0).standard_normal((50, 6)).astype(np.float32)
    assert regression_gate(a, a.copy(), cos_min=0.99, mse_max=1e-3)["passed"] is True
    b = a + 5.0
    assert regression_gate(a, b, cos_min=0.99, mse_max=1e-3)["passed"] is False
