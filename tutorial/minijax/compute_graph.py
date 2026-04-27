import itertools
from dataclasses import dataclass

from .core import InterpreterABC, ValueABC, Primitive, set_interpreter


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
    # outvar = add invar1 invar2
    primitive: Primitive
    invars: tuple[Var, ...]
    outvar: Var

    def __repr__(self) -> str:
        repr = f"{self.outvar} = {self.primitive.name} "
        return repr + " ".join(map(str, self.invars))


_var_ids = itertools.count()


class Var:
    def __init__(self):
        self._id = next(_var_ids)

    def __repr__(self) -> str:
        def letters(i):
            return chr(97 + i) if i < 26 else letters(i // 26 - 1) + chr(97 + (i % 26))

        return f"{letters(self._id)}"


class MakeCG(InterpreterABC):
    def __init__(self):
        self.equations = []

    def process_primitive(self, primitive, *args, **options):
        invars = [a.var for a in args]
        outvar = Var()
        eqn = Equation(primitive, tuple(invars), outvar)
        self.equations.append(eqn)
        return AbstractValue(outvar)


@dataclass
class AbstractValue(ValueABC):
    var: Var


def make_compute_graph(fn, *args) -> ComputeGraph:
    interpreter = MakeCG()
    set_interpreter(interpreter)
    invars = [Var() for _ in args]
    invals = [AbstractValue(v) for v in invars]

    outvals = fn(*invals)
    if isinstance(outvals, AbstractValue):
        outvals = (outvals,)

    outvars = [av.var for av in outvals]
    return ComputeGraph(invars, outvars, interpreter.equations)

