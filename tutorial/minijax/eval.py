from dataclasses import dataclass

import numpy as np

from .core import InterpreterABC, ValueABC, neg, add, mul, matmul, relu, transpose, relu_derivative


class EvalInterpreter(InterpreterABC):
    def process_primitive(self, primitive, *args, **options):
        args = [a.value for a in args]
        rule = eval_rules[primitive]
        res = rule(*args, **options)
        return Array(res)


@dataclass
class Array(ValueABC):
    value: np.ndarray

    @property
    def shape(self):
        return self.value.shape


def np_dot(x, y):  # np.dot doesn't broadcast
    if y.ndim <= 1:
        return np.dot(x, y)
    return np.einsum("...j,...jk", x, y)


eval_rules = {
    neg: lambda x: -x,
    add: lambda x, y: x + y,
    mul: lambda x, y: x * y,
    matmul: np_dot,
    relu: lambda x: np.maximum(0.0, x),
    transpose: lambda x: np.transpose(x),
    relu_derivative: lambda x: np.where(x > 0.0, 1.0, 0.0),
}
