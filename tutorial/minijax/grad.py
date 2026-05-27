import numpy as np

from .core import neg, add, mul, matmul, relu, transpose, relu_derivative
from .eval import Array
from .compute_graph import make_compute_graph


def grad(fn, *primals):
    out_tangent = Array(np.array(1.0))
    return vjp(fn, primals, (out_tangent,))


def vjp(fn, primals, out_tangents):
    cg = make_compute_graph(fn, *primals)
    print(cg)

    inner_primals = _forward(cg, primals)
    return _backward(cg, inner_primals, out_tangents)


def _forward(cg, primals):
    env = {}

    for in_var, in_val in zip(cg.invars, primals):
        env[in_var] = in_val

    for eqn in cg.equations:
        args = [env[iv] for iv in eqn.invars]
        res = eqn.primitive(*args)
        env[eqn.outvar] = res

    return env


def _backward(cg, inner_primals, out_tangents):
    tangents = {}

    def update(v, tangent):
        if v in tangents:
            tangents[v] = tangents[v] + tangent
        else:
            tangents[v] = tangent

    for ov, tangent in zip(cg.outvars, out_tangents):
        update(ov, tangent)

    for eqn in reversed(cg.equations):
        out_tangent = tangents[eqn.outvar]
        in_primals = [inner_primals[iv] for iv in eqn.invars]

        rule = vjp_rules[eqn.primitive]
        print(
            eqn.primitive,
            [iv.shape for iv in eqn.invars],
            eqn.outvar.shape,
            [iv.shape for iv in in_primals],
            out_tangent.shape,
        )
        in_tangents = rule(out_tangent, *in_primals)
        print([t.shape for t in in_tangents])

        for iv, in_tangent in zip(eqn.invars, in_tangents):
            update(iv, in_tangent)

    return tuple(tangents[in_var] for in_var in cg.invars)


vjp_rules = {
    neg: lambda g, x: (-g,),
    add: lambda g, x, y: (g, g),
    # mul: lambda x, y: x * y,
    transpose: lambda g, x: (transpose(g),),
    matmul: lambda g, x, y: (matmul(transpose(y), g), (matmul(g, transpose(x)))),
    relu: lambda g, x: (relu_derivative(x) * g,),
}
