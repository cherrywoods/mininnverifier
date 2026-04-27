from minijax.eval import Array, EvalInterpreter
from minijax.core import relu, set_interpreter
from minijax.compute_graph import make_compute_graph


def nn(x):
    y = Array(2) * x
    return relu(y) 

def nn2(x, y):
    z = y * x
    return relu(z) 

# set_interpreter(EvalInterpreter())
# y = nn(Array(-5.0))
# print(f"{y=}")

cg = make_compute_graph(nn2, Array(2.0), Array(10.0))
print(cg)
