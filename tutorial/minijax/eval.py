from dataclasses import dataclass

from .core import InterpreterABC, ValueABC, neg, add, mul, relu


class EvalInterpreter(InterpreterABC):
    def process_primitive(self, primitive, *args, **options):
        print(args)
        args = [a.value for a in args]
        rule = eval_rules[primitive]
        res = rule(*args, **options)
        return Array(res)


@dataclass
class Array(ValueABC):
    value: float


eval_rules = {
    neg: lambda x: -x,
    add: lambda x, y: x + y,
    mul: lambda x, y: x * y,
    relu: lambda x: max(0, x),
}
