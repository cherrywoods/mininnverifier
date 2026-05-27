# Copyright (c) 2025 by David Boetius
# Licensed under the MIT Licensed.
from dataclasses import dataclass

from minijax import core
from minijax.nested_containers import flatten, map_structure
from minijax.eval import Array


@dataclass
class Box:
    lb: core.Value
    ub: core.Value


def _is_ibp_leaf(x):
    return isinstance(x, (Box, core.Value))


def ibp(fn):
    def ibp_fn(*args: Box | core.Value, **kwargs):
        flat_args = flatten(args, is_leaf=_is_ibp_leaf)[0]
        level_values = [a.lb if isinstance(a, Box) else a for a in flat_args]
        interpreter = core.new_interpreter(IBPInterpreter, level_values)
        vals = map_structure(interpreter.wrap, args, is_leaf=_is_ibp_leaf)

        out_bounds = fn(*vals, **kwargs)
        return map_structure(lambda ibp_val: Box(ibp_val.lb, ibp_val.ub), out_bounds)

    return ibp_fn


class IBPValue(core.Value):
    def __init__(self, interpreter, lb, ub, is_point=False):
        super().__init__(interpreter, lb.shape)
        self.lb = lb  # lower bound
        self.ub = ub  # upper bound
        self.is_point = is_point  # whether lb == ub


class IBPInterpreter(core.Interpreter[IBPValue]):
    def __init__(self, level: int):
        super().__init__(level)

    def wrap(self, value):
        if isinstance(value, IBPValue):
            return value
        elif isinstance(value, Box):
            return IBPValue(self, value.lb, value.ub)
        return IBPValue(self, value, value, is_point=True)

    def process(self, primitive, values, options):
        if all(box.is_point for box in values):
            res = primitive(*[box.lb for box in values], **options)
            return IBPValue(self, res, res, is_point=True)

        if primitive in mono_non_dec_primitives:
            out_lb, out_ub = ibp_monotonic_non_decreasing(primitive, *values, **options)
        elif primitive in mono_non_inc_primitives:
            out_lb, out_ub = ibp_monotonic_non_increasing(primitive, *values, **options)
        elif primitive in linear_primitives:
            out_lb, out_ub = ibp_linear(primitive, *values, **options)
        else:
            raise NotImplementedError(f"No IBP rule for primitive {primitive}")
        return IBPValue(self, out_lb, out_ub)


def ibp_monotonic_non_decreasing(fn, *args, **options):
    in_lbs, in_ubs = [box.lb for box in args], [box.ub for box in args]
    out_lb = fn(*in_lbs, **options)
    out_ub = fn(*in_ubs, **options)
    return out_lb, out_ub


def ibp_monotonic_non_increasing(fn, *args, **options):
    out_ub, out_lb = ibp_monotonic_non_decreasing(fn, *args, **options)
    return out_lb, out_ub


def ibp_linear(fn, *args, **options):
    x, y = args
    if not x.is_point and not y.is_point:
        raise NotImplementedError(f"No IBP rule for bilinear application of primitive {fn}")
    elif x.is_point:
        x = x.lb
        y_mid = (y.ub + y.lb) * Array(0.5)
        y_ran = (y.ub - y.lb) * Array(0.5)
        out_mid = fn(x, y_mid)
        out_ran = fn(core.abs(x), y_ran)
        return out_mid - out_ran, out_mid + out_ran
    elif y.is_point:
        return ibp_linear(lambda y, x: fn(x, y, **options), y, x)


mono_non_dec_primitives = {
    core.expand_dims,
    core.moveaxis,
    core.reshape,
    core.add,
    core.reduce_sum,
    core.relu,
    core.exp,
}
mono_non_inc_primitives = {core.neg}
linear_primitives = {core.dot, core.mul}
