# Copyright (c) 2025 by David Boetius
# Licensed under the MIT License.
"""Evaluate a mininn network on binary input files.

Usage:
    eval --output-dir <dir> <network.mininn> <input1.bin> [<input2.bin> ...]
_forward
Each input file contains float64 values in row-major order matching the shape
of the corresponding network input variable. Outputs are written as float64
binary files to the output directory, one file per output variable.
Filenames are printed to stdout, one per line.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from minijax.serialize import load
from minijax.eval import Array
from minijax.jit import run_graph


def main():
    parser = argparse.ArgumentParser(description="Evaluate a mininn network.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("network", type=str)
    parser.add_argument("inputs", nargs="*", type=str)
    args = parser.parse_args()

    graph = load(args.network)

    if len(args.inputs) != len(graph.invars):
        print(
            f"Error: network has {len(graph.invars)} input(s), "
            f"but {len(args.inputs)} input file(s) were provided.",
            file=sys.stderr,
        )
        sys.exit(1)

    inputs = [
        Array(np.fromfile(path, dtype=np.float64).reshape(var.shape))
        for var, path in zip(graph.invars, args.inputs)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = run_graph(graph, inputs)
    for i, out in enumerate(outputs):
        out_path = args.output_dir / f"output_{i}.bin"
        out.array.tofile(out_path)
        print(out_path)


if __name__ == "__main__":
    main()
