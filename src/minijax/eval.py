# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
import numpy as np

from . import core


class Array(core.Value):
    def __init__(self, array_like):
        self.array = np.asarray(array_like)
        super().__init__(EvalInterpreter(), self.array.shape)

    def item(self):
        return self.array.item()

    def __repr__(self):
        data_str = str(self.array).replace("\n", "\n" + " " * len("Array("))
        return f"Array({data_str})"


def full(shape, fill_value):
    return Array(np.full(shape, fill_value, dtype=np.float64))


def zeros(shape):
    return full(shape, 0.0)


def ones(shape):
    return full(shape, 1.0)


def broadcast_to(value, shape):
    # exploiting that add does broadcasting
    return core.add(value, zeros(shape))


class EvalInterpreter(core.Interpreter[Array]):
    def wrap(self, value):
        if not isinstance(value, core.Value):
            return Array(value)
        elif not isinstance(value, Array):
            raise ValueError("EvalInterpreter must be the bottom interpreter")
        return value

    def process(self, primitive, values: list[Array], options: dict):
        np_vals = [v.array for v in values]
        np_out = eval_rules[primitive](*np_vals, **options)
        return Array(np_out)


def np_dot(x, y):  # np.dot doesn't broadcast
    if x.ndim == 0 or y.ndim == 0:
        return x * y
    elif y.ndim <= 1 or x.ndim <= 1:
        return np.dot(x, y)
    return np.einsum("...ij,...jk->...ik", x, y)


eval_rules = {
    core.expand_dims: lambda x, axes: np.expand_dims(x, axes),
    core.moveaxis: np.moveaxis,
    core.reshape: lambda x, new_shape: np.reshape(x, new_shape),
    core.neg: lambda x: -x,
    core.add: lambda x, y: x + y,
    core.reduce_sum: lambda x, axes: x.sum(tuple(axes)),
    core.dot: np_dot,
    core.mul: lambda x, y: x * y,
    core.reciprocal: lambda x: 1 / x,
    core.relu: lambda x: np.maximum(x, 0.0),
    core.square: np.square,
    core.sqrt: np.sqrt,
    core.exp: np.exp,
    core.log: np.log,
    core.where: np.where,
    core.greater_equal: np.greater_equal,
    core.less_equal: np.less_equal,
    core.elementwise_not: np.bitwise_not,
    core.elementwise_and: np.bitwise_and,
    core.concat_two: lambda x, y, axis: np.concatenate([x, y], axis),
    core.head: lambda x, axis, index: np.split(x, [index], axis)[0],
    core.tail: lambda x, axis, index: np.split(x, [index], axis)[1],
}
