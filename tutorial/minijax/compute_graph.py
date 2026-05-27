import itertools
from dataclasses import dataclass

import numpy as np

from .core import InterpreterABC, ValueABC, Primitive, push_interpreter, pop_interpreter
from .core import neg, add, mul, matmul, relu, transpose


@dataclass(frozen=True)
class ComputeGraph:
    invars: tuple[Var, ...]
    outvars: tuple[Var, ...]
    equations: tuple[Equation, ...]

    def __repr__(self) -> str:
        repr = "input: " + " ".join(map(str, self.invars)) + "\n"
        repr += "\n".join([f"  {eqn}" for eqn in self.equations]) + "\n"
        return repr + "output: " + " ".join(map(str, self.outvars))


@dataclass(frozen=True)
class Equation:
    primitive: Primitive
    invars: tuple[Var, ...]
    outvar: Var

    def __repr__(self) -> str:
        repr = f"{self.outvar} = {self.primitive.name} "
        return repr + " ".join(map(str, self.invars))


_var_ids = itertools.count()


class Var:
    def __init__(self, shape):
        self._id = next(_var_ids)
        self.shape = shape

    def __repr__(self) -> str:
        def letters(i):
            return chr(97 + i) if i < 26 else letters(i // 26 - 1) + chr(97 + (i % 26))

        return f"{letters(self._id)}{[self.shape]}"


class MakeCG(InterpreterABC):
    def __init__(self):
        self.equations = []

    def process_primitive(self, primitive, *args, **options):
        invars = [a.var for a in args]
        out_shape = shape_rules[primitive](*[iv.shape for iv in invars], **options)
        outvar = Var(out_shape)
        eqn = Equation(primitive, tuple(invars), outvar)
        self.equations.append(eqn)
        return AbstractValue(outvar)


@dataclass
class AbstractValue(ValueABC):
    var: Var

    @property
    def shape(self):
        return self.var.shape


def matmul_shape_rule(x_shape, y_shape):
    out_shape = ()
    if len(x_shape) > 1:
        out_shape += (x_shape[0],)
    elif len(y_shape) > 1:
        out_shape += (y_shape[1],)
    return out_shape


shape_rules = {
    neg: lambda x_shape: x_shape,
    relu: lambda x_shape: x_shape,
    add: lambda x_shape, y_shape: np.broadcast_shapes(x_shape, y_shape),
    mul: lambda x_shape, y_shape: np.broadcast_shapes(x_shape, y_shape),
    transpose: lambda x_shape: tuple(reversed(x_shape)),
    matmul: matmul_shape_rule,
}


def make_compute_graph(fn, *args) -> ComputeGraph:
    interpreter = MakeCG()
    push_interpreter(interpreter)
    invars = [Var(a.shape) for a in args]
    invals = [AbstractValue(v) for v in invars]

    outvals = fn(*invals)
    if isinstance(outvals, AbstractValue):
        outvals = (outvals,)

    outvars = [av.var for av in outvals]
    pop_interpreter()
    return ComputeGraph(invars, outvars, interpreter.equations)
