# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
from dataclasses import dataclass

from minijax import core
from minijax.core import Value, abs, where, relu
from minijax.nested_containers import map_structure
from minijax.eval import Array, zeros


@dataclass
class Box:
    lb: core.Value
    ub: core.Value


def box_or_val(obj):
    return isinstance(obj, (Box, core.Value))


def ibp(fn):
    def ibp_fn(*args: Box | core.Value, **kwargs):
        with core.new_interpreter(IBPInterpreter()) as interpreter:
            vals = map_structure(interpreter.wrap, args, is_leaf=box_or_val)
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
    def wrap(self, value):
        if isinstance(value, IBPValue):
            return value
        elif isinstance(value, Box):
            return IBPValue(self, value.lb, value.ub)
        if not isinstance(value, core.Value):
            value = Array(value)
        return IBPValue(self, value, value, is_point=True)

    def process(self, primitive, values, options):
        if all(v.is_point for v in values):
            res = primitive(*[v.lb for v in values], **options)
            return IBPValue(self, res, res, is_point=True)

        if primitive in mono_non_dec_primitives:
            out_lb, out_ub = ibp_monotonic_non_decreasing(primitive, *values, **options)
        elif primitive in mono_non_inc_primitives:
            out_lb, out_ub = ibp_monotonic_non_increasing(primitive, *values, **options)
        elif primitive in linear_primitives:
            out_lb, out_ub = ibp_linear(primitive, *values, **options)
        elif primitive is core.square:
            out_lb, out_ub = ibp_square(*values, **options)
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
    return out_lb, out_ub  # swaped from ibp_monotonic_non_decreasing


def ibp_linear(fn, x, y, **options):
    if not x.is_point and not y.is_point:
        raise NotImplementedError(f"No IBP rule for bilinear application of primitive {fn}")
    elif x.is_point:
        x = x.lb
        y_mid = (y.ub + y.lb) * 0.5
        y_ran = (y.ub - y.lb) * 0.5
        out_mid = fn(x, y_mid, **options)
        out_ran = fn(abs(x), y_ran, **options)
        return out_mid - out_ran, out_mid + out_ran
    elif y.is_point:
        return ibp_linear(lambda y, x: fn(x, y, **options), y, x)


def ibp_square(x):
    y_l, y_r = core.square(x.lb), core.square(x.ub)
    # x.lb >= 0 => monotonic increasing
    # x.ub <= 0 => monotonic decreasing
    # x.lb < 0 < x.ub => lb = 0.0, ub = max(x.lb^2 , x.ub^2)
    y_lb = where(x.lb >= 0.0, y_l, where(x.ub < 0.0, y_r, zeros(x.shape)))
    # x.ub > -x.lb => x.ub + x.lb > 0
    y_ub = where(x.lb >= 0.0, y_r, where(x.ub < 0.0, y_l, where(-x.lb >= x.ub, y_l, y_r)))
    return y_lb, y_ub


mono_non_dec_primitives = {
    core.expand_dims,
    core.moveaxis,
    core.reshape,
    core.concat,
    core.head,
    core.tail,
    core.add,
    core.reduce_sum,
    core.relu,
    core.exp,
}
mono_non_inc_primitives = {core.neg}
linear_primitives = {core.dot, core.mul}
