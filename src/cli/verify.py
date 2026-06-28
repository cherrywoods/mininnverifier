# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Verify a property of a mininn network via input-splitting branch-and-bound.

Usage:
    verify --output-dir <dir> <network.mininn> box <lb.bin> <ub.bin>

The property is *baked into the network*: its single scalar output is a *margin*
that we verify satisfies ``margin(x) >= 0`` for all ``x`` in the input box. Input
specs use the same inline markers as ``bounds``:

    box   <lb.bin> <ub.bin>     interval input (the verification variable)
    point <value.bin>           fixed input (a constant during verification)

Exactly one ``box`` input is expected (the variable that branch-and-bound splits);
any number of ``point`` inputs may also be given.

Output (stdout):
    Arbitrary progress logging may appear first. The *final* line is the verdict:

        sat     the property holds (margin >= 0 everywhere in the box)
        viol    the property is violated; a counterexample was found

    When the verdict is ``viol``, the counterexample is written to
    ``<output-dir>/counterexample_<i>.bin`` (one file per network input) and the
    path(s) are printed on the line(s) immediately *before* the verdict.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from minijax.serialize import load
from minijax.jit import run_graph
from mininnverifier.ibp import Box
from mininnverifier.input_splitting_bab import input_splitting_bab

from cli.bounds import _parse_inputs


def main():
    parser = argparse.ArgumentParser(
        description="Verify margin(x) >= 0 over an input box via input-splitting BaB."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("network", type=str)
    parser.add_argument("inputs", nargs="*", type=str)
    args = parser.parse_args()

    graph = load(args.network)
    inputs = _parse_inputs(args.inputs, graph.invars)

    box_positions = [i for i, inp in enumerate(inputs) if isinstance(inp, Box)]
    if len(box_positions) != 1:
        print(
            f"Error: verify expects exactly one 'box' input (the verification "
            f"variable), got {len(box_positions)}.",
            file=sys.stderr,
        )
        sys.exit(1)
    box_pos = box_positions[0]
    x_box = inputs[box_pos]

    def margin(x):
        full_inputs = list(inputs)
        full_inputs[box_pos] = x
        return run_graph(graph, full_inputs)[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        counterexample = input_splitting_bab(margin)(x_box)
    except RuntimeError as exc:
        # Could not decide within the branch-and-bound budget (IBP too loose).
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    if counterexample is None:
        print("sat")
        return

    # Violated: assemble the full witness (counterexample for the box input,
    # the fixed value for each point input), write one .bin per network input,
    # print the paths, and finish with the verdict on the final line.
    witness = list(inputs)
    witness[box_pos] = counterexample
    for i, inp in enumerate(witness):
        cx_path = args.output_dir / f"counterexample_{i}.bin"
        np.asarray(inp.array, dtype=np.float64).tofile(cx_path)
        print(cx_path)
    print("viol")


if __name__ == "__main__":
    main()
