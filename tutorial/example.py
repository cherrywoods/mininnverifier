import numpy as np

from minijax.eval import Array, EvalInterpreter
from minijax.core import relu, matmul, push_interpreter
from minijax.compute_graph import make_compute_graph
from minijax.ibp import DirectIntervalEvalInterpreter, Box
from minijax.vmap import VMapInterpreter, VmappedArray
from minijax.grad import grad

n = 3
m = 2


def nn(x):
    w = (np.arange(m * n) - 3.0).reshape((m, n))
    print(f"{w=}")
    z = matmul(Array(w), x)
    print(f"{z=}")
    return relu(z)


def nn2(x, y):
    z = y + x
    return relu(z)


def nn3(x):
    z = x + x
    return relu(z)


def nn4(x, w1, w2):
    z = matmul(x, w1)
    z = relu(z)
    return matmul(z, w2)


print("-" * 100)

# push_interpreter(EvalInterpreter())
# push_interpreter(VMapInterpreter())
#
# x = np.array([
#     [0.0, 0.0, -5.0],
#     [1.0, 0.0, -1.0]
# ])
# x = Array(x)
# x = VmappedArray(0, x)
# print(f"{x=}")
# y = nn(x)
# print(f"{y=}")


# push_interpreter(EvalInterpreter())
# push_interpreter(VMapInterpreter())
# push_interpreter(DirectIntervalEvalInterpreter())
#
# x_lb = np.array([
#     [1.0, 0.0, -1.0],
#     [-1.0, -1.0, -10.0],
# ])
# x_ub = np.array([
#     [2.0, 1.0, 1.0],
#     [0.0, 0.0, 0.0]
# ])
# x_lb, x_ub = Array(x_lb), Array(x_ub)
# x_lb, x_ub = VmappedArray(0, x_lb), VmappedArray(0, x_ub)
# print(f"{x_lb=}")
# print(f"{x_ub=}")
# y = nn3(Box(x_lb, x_ub))
# print(f"{y=}")

# cg = make_compute_graph(nn2, Array(2.0), Array(10.0))
# print(cg)

push_interpreter(EvalInterpreter())

x = np.array([[0.0, 0.0, -5.0], [1.0, 0.0, -1.0]])
x = Array(x)
w1 = Array(np.arange(3) - 1.5)
w2 = Array(np.array([-1.0, 1.0]))
y = grad(nn4, x, w1, w2)
print(y)

print("-" * 100)
