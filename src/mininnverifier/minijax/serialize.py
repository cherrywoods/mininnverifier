# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
"""Import and export compute graphs.

The compute graphs are stored in a .zip file containing
a `graph.txt` file with the graph equations and an arbitrary
number of binary files containing the constant values.
For the constant `xyz`, the file `xyz.bin` contains the float64
values of the constant in row-major order. The shape of the constant
is specified in the `graph.txt` file.

Variables are lower case, constants are upper case.
"""

from pathlib import Path
import zipfile

import numpy as np

from . import core as minijax_core
from .compute_graph import ComputeGraph, Const, Equation, Var


def dump(graph: ComputeGraph, file: str | Path):
    """Export a compute graph to a .zip file."""
    with zipfile.ZipFile(file, "w") as zf:
        graph_str, consts = _serialize_graph(graph)
        with zf.open("graph.txt", "w") as f:
            f.write(graph_str.encode("utf-8"))

        consts = {name: const.value.astype(np.float64) for name, const in consts.items()}
        const_bytes = {name: val.tobytes("c") for name, val in consts.items()}
        for name, bytes in const_bytes.items():
            with zf.open(f"{name}.bin", "w") as f:
                f.write(bytes)


def load(file: str) -> ComputeGraph:
    with zipfile.ZipFile(file, "r") as zf:
        graph_str = zf.read("graph.txt").decode("utf-8")
        consts = {}
        for file in zf.namelist():
            if file.endswith(".bin"):
                name = file[:-4]
                consts[name] = zf.read(file)
    return _deserialize_graph(graph_str, consts)


def _serialize_graph(graph) -> tuple[str, dict[str, Const]]:
    var_ids = {}
    const_ids = {}

    def letters(i):
        return chr(97 + i) if i < 26 else letters(i // 26 - 1) + chr(97 + (i % 26))

    def serialize_atom(atom) -> str:
        id_dict = var_ids if isinstance(atom, Var) else const_ids
        if atom not in id_dict:
            id_dict[atom] = len(id_dict)
        i = id_dict[atom]
        name = letters(i)
        if isinstance(atom, Const):
            name = name.upper()
        return name + "[" + ", ".join(map(str, atom.shape)) + "]"

    def serialize_eqn(eqn):
        opts = "{" + ", ".join([f"{k}: {v}" for k, v in eqn.options.items()]) + "}"
        repr = f"{serialize_atom(eqn.outvar)} = {eqn.primitive.name}{opts} "
        return repr + " ".join(map(serialize_atom, eqn.inputs))

    out = "input: " + ", ".join(map(serialize_atom, graph.invars)) + "\n"
    out += "\n".join(map(serialize_eqn, graph.equations)) + "\n"
    out += "output: " + ", ".join(map(serialize_atom, graph.outvars))

    consts = {letters(i).upper(): const for const, i in const_ids.items()}
    return out, consts


def _deserialize_graph(graph_repr: str, consts: dict[str, bytes]) -> ComputeGraph:
    lines = graph_repr.splitlines()
    input_line = lines[0].strip()
    output_line = lines[-1].strip()
    atoms = {}

    def deserialize_atom(repr: str):
        name, shape_str = repr.split("[")
        name = name.strip()
        shape = shape_str[:-1].split(",")
        shape = tuple(int(s.strip()) for s in shape)

        if name not in atoms:
            if name.upper() == name:  # uppercase => Const
                val_bytes = consts[name]
                val = np.frombuffer(val_bytes).reshape(shape)
                atoms[name] = Const(val)
            else:
                atoms[name] = Var(shape)

        atom = atoms[name]
        assert atom.shape == shape, f"Inconsistent shapes for {name}: {atom.shape}, {shape}."
        return atom

    def deserialize_eqn(repr: str):
        outvar, expr = repr.split("=")
        primitive, *inputs = expr.strip().split()
        primitive, opts = primitive.split("{")

        primitive = getattr(minijax_core, primitive)
        opts = opts[:-1].strip().split(",")
        opts = [opt.split(":") for opt in opts]
        opts = {key.strip(): val.strip() for key, val in opts}

        _, outvar = deserialize_atom(outvar)
        inputs = tuple(deserialize_atom(iv)[1] for iv in inputs)
        return Equation(primitive, inputs, outvar, opts)

    invars = input_line[len("input:") :].split(",")
    invars = tuple(map(deserialize_atom, invars))
    eqns = tuple(map(deserialize_eqn, lines))
    outvars = output_line[len("output:") :].split(",")
    outvars = tuple(map(deserialize_atom, outvars))
    return ComputeGraph(invars, outvars, eqns)
