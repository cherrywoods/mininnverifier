#!/usr/bin/env python3
# Copyright (c) 2026 by David Boetius
# Licensed under the MIT License.
"""Generate milestone2 bounds tests by reusing milestone1 base/unit/eval networks.

For each milestone1 eval test that uses only IBP-supported primitives we emit
two milestone2 test directories — one under ``base/unit/bounds`` (with a small
sample set and a reference IBP bound the impl is checked against with a 1.5x
tightness margin) and one under ``base/fuzz/bounds`` (with many random samples
and no tightness check; soundness only).

The IBP reference walker in this script is a *separate* numpy implementation of
the canonical IBP rules — it does not rely on ``mininnverifier.ibp``.

Run from the repository root:
    python tests/milestone2/generate_bounds_tests.py
"""

import json
import shutil
from pathlib import Path

import numpy as np

from minijax.eval import Array
from minijax.jit import run_graph
from minijax.serialize import load


SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MS1_EVAL = REPO_ROOT / "tests/milestone1/base/unit/eval"
MS1_OPEN_FUZZ_EVAL = REPO_ROOT / "tests/milestone1/open/fuzz/eval"
UNIT_OUT = SCRIPT_DIR / "base/unit/bounds"
FUZZ_OUT = SCRIPT_DIR / "base/fuzz/bounds"
OPEN_FUZZ_OUT = SCRIPT_DIR / "open/fuzz/bounds"

# TODO: port the milestone1 open/unit/eval tests to milestone2/open/unit/bounds.
# Most of those networks use primitives that the current minijax loader can't
# deserialize (avgpool, conv, gelu, pad, ...), so the test fixtures cannot
# include reference IBP bounds or pre-evaluated sample outputs.  A pragmatic
# port would copy each .mininn opaquely, perturb input bins flat (input ± EPS),
# mark every input as "box", and use a lenient structural check (e.g. one that
# verifies the impl produced finite lb/ub pairs).  Most tests will then fail
# at command level under the current impl; a few will pass structurally.
# Picking up box @ box support in ibp.py is a prerequisite for a meaningful
# correctness check on open tests with transformers.

EPS = 0.05
N_UNIT_SAMPLES = 16
N_FUZZ_SAMPLES = 512
TIGHTNESS_FACTOR = 1.5
SAMPLE_ATOL = 1e-9
RNG_UNIT_SEED = 42
RNG_FUZZ_SEED = 43

LINEAR_PRIMS = {"dot", "mul"}
IBP_SUPPORTED = {
    "add", "expand_dims", "moveaxis", "reshape", "reduce_sum", "relu", "exp",
    "neg", "dot", "mul",
}


# ---------------------------------------------------------------------------
# Numpy reference IBP walker.
# ---------------------------------------------------------------------------


def _np_dot(x, y):
    if y.ndim <= 1:
        return np.dot(x, y)
    return np.einsum("...j,...jk", x, y)


# kind: "mnd" = monotonic non-decreasing in every argument
#       "mni" = monotonic non-increasing in every argument
#       "lin" = bilinear, one operand must be a point
NP_RULES = {
    "add":         (np.add, "mnd"),
    "neg":         (np.negative, "mni"),
    "relu":        (lambda x: np.maximum(x, 0.0), "mnd"),
    "exp":         (np.exp, "mnd"),
    "expand_dims": (lambda x, axes: np.expand_dims(x, axes), "mnd"),
    "moveaxis":    (lambda x, source, destination: np.moveaxis(x, source, destination), "mnd"),
    "reshape":     (lambda x, new_shape: x.reshape(new_shape), "mnd"),
    "reduce_sum":  (lambda x, axes: x.sum(axis=axes), "mnd"),
    "mul":         (np.multiply, "lin"),
    "dot":         (_np_dot, "lin"),
}


def reference_ibp(graph, input_boxes):
    """Walk *graph* applying numpy IBP rules.

    ``input_boxes`` is a list of ``(lb_np, ub_np)`` per graph invar; supply
    ``lb == ub`` for point inputs. Returns a list of ``(out_lb, out_ub)`` per
    output variable.
    """
    env = {
        v: (np.asarray(lb), np.asarray(ub))
        for v, (lb, ub) in zip(graph.invars, input_boxes)
    }

    for eq in graph.equations:
        in_boxes = []
        for atom in eq.inputs:
            if atom.is_const:
                v = atom.value.array
                in_boxes.append((v, v))
            else:
                in_boxes.append(env[atom])

        prim_name = eq.primitive.name
        fn, kind = NP_RULES[prim_name]
        opts = dict(eq.options)

        if kind == "mnd":
            lbs = [b[0] for b in in_boxes]
            ubs = [b[1] for b in in_boxes]
            out_lb = fn(*lbs, **opts)
            out_ub = fn(*ubs, **opts)
        elif kind == "mni":
            lbs = [b[0] for b in in_boxes]
            ubs = [b[1] for b in in_boxes]
            out_lb = fn(*ubs, **opts)
            out_ub = fn(*lbs, **opts)
        elif kind == "lin":
            x, y = in_boxes
            x_pt = np.array_equal(x[0], x[1])
            y_pt = np.array_equal(y[0], y[1])
            if x_pt and y_pt:
                v = fn(x[0], y[0], **opts)
                out_lb = out_ub = v
            elif y_pt:
                xm = 0.5 * (x[0] + x[1])
                xr = 0.5 * (x[1] - x[0])
                m = fn(xm, y[0], **opts)
                r = fn(xr, np.abs(y[0]), **opts)
                out_lb, out_ub = m - r, m + r
            elif x_pt:
                ym = 0.5 * (y[0] + y[1])
                yr = 0.5 * (y[1] - y[0])
                m = fn(x[0], ym, **opts)
                r = fn(np.abs(x[0]), yr, **opts)
                out_lb, out_ub = m - r, m + r
            else:
                raise NotImplementedError(
                    f"both operands of {prim_name} are boxes — current IBP "
                    f"does not support box @ box"
                )
        else:
            raise NotImplementedError(kind)

        env[eq.outvar] = (out_lb, out_ub)

    return [env[v] for v in graph.outvars]


# ---------------------------------------------------------------------------
# Fixture generation.
# ---------------------------------------------------------------------------


def _decide_kinds(graph):
    """Per-invar "box" / "point" assignment.

    * 1 invar  -> "box"
    * 2 invars on a linear primitive (dot/mul) -> the second is "point"
      (avoids the unsupported box @ box case)
    * 2+ invars on a non-linear primitive -> all "box"
    """
    n = len(graph.invars)
    if n == 1:
        return ["box"]
    if n == 2 and graph.equations:
        prim0 = graph.equations[0].primitive.name
        if prim0 in LINEAR_PRIMS:
            return ["box", "point"]
    return ["box"] * n


def _read_ms1_input(ms1_dir, idx, shape):
    return np.fromfile(
        ms1_dir / "resources" / f"input_{idx}.bin", dtype=np.float64
    ).reshape(shape)


def _eval_at_point(graph, inputs):
    arrays = [Array(np.asarray(x, dtype=np.float64)) for x in inputs]
    return [o.array for o in run_graph(graph, arrays)]


def _build_fixtures(ms1_dir, graph, kinds, n_samples, rng):
    centers = []
    boxes = []
    for i, (var, kind) in enumerate(zip(graph.invars, kinds)):
        c = _read_ms1_input(ms1_dir, i, var.shape)
        centers.append(c)
        if kind == "box":
            boxes.append((c - EPS, c + EPS))
        else:
            boxes.append((c, c))

    ref_bounds = reference_ibp(graph, boxes)
    out_shapes = [v.shape for v in graph.outvars]

    # Samples: first the midpoint, then random.
    samples = [list(centers)]
    for _ in range(n_samples - 1):
        s = []
        for i, kind in enumerate(kinds):
            if kind == "box":
                s.append(rng.uniform(boxes[i][0], boxes[i][1]))
            else:
                s.append(centers[i])
        samples.append(s)

    sample_outs = [[] for _ in graph.outvars]
    for s in samples:
        outs = _eval_at_point(graph, s)
        for j, o in enumerate(outs):
            sample_outs[j].append(o)
    sample_outs = [np.stack(arrs, axis=0) for arrs in sample_outs]

    return centers, boxes, ref_bounds, sample_outs, out_shapes


def _emit_test(
    out_root, name, ms1_dir, graph_path, kinds, centers, boxes,
    ref_bounds, sample_outs, out_shapes, *, with_reference,
):
    test_dir = out_root / name
    if test_dir.exists():
        shutil.rmtree(test_dir)
    res_dir = test_dir / "resources"
    res_dir.mkdir(parents=True)

    network_name = graph_path.name
    shutil.copy2(graph_path, res_dir / network_name)

    input_tokens = []
    for i, kind in enumerate(kinds):
        if kind == "box":
            lb_path = f"resources/input_{i}_lb.bin"
            ub_path = f"resources/input_{i}_ub.bin"
            boxes[i][0].astype(np.float64).tofile(test_dir / lb_path)
            boxes[i][1].astype(np.float64).tofile(test_dir / ub_path)
            input_tokens += ["box", lb_path, ub_path]
        else:
            p = f"resources/input_{i}.bin"
            centers[i].astype(np.float64).tofile(test_dir / p)
            input_tokens += ["point", p]

    sample_out_paths = []
    for j, arr in enumerate(sample_outs):
        p = f"resources/sample_outputs_{j}.bin"
        arr.astype(np.float64).tofile(test_dir / p)
        sample_out_paths.append(p)

    config = {
        "command": "bounds",
        "network": f"resources/{network_name}",
        "inputs": input_tokens,
        "check": "bounds_within_range",
        "output_shapes": [list(s) for s in out_shapes],
        "sample_outputs": sample_out_paths,
        "sample_atol": SAMPLE_ATOL,
    }

    if with_reference:
        ref_lb_paths, ref_ub_paths = [], []
        for j, (lb, ub) in enumerate(ref_bounds):
            lp = f"resources/reference_output_{j}_lb.bin"
            up = f"resources/reference_output_{j}_ub.bin"
            lb.astype(np.float64).tofile(test_dir / lp)
            ub.astype(np.float64).tofile(test_dir / up)
            ref_lb_paths.append(lp)
            ref_ub_paths.append(up)
        config["reference_lb"] = ref_lb_paths
        config["reference_ub"] = ref_ub_paths
        config["tightness_factor"] = TIGHTNESS_FACTOR

    (test_dir / "test.json").write_text(json.dumps(config, indent=4) + "\n")
    print(f"  created {test_dir.relative_to(REPO_ROOT)}")


def _port_open_fuzz_test(src_test_json, dest_dir):
    """Convert a milestone1 ``fuzz_eval`` config to a milestone2 ``fuzz_bounds``
    config — same trial count, seed, and primitive set, but the bounds CLI is
    exercised with each invar widened into an axis-aligned box.
    """
    cfg = json.loads(src_test_json.read_text())
    new_cfg = {
        "command": "fuzz_bounds",
        "n_trials": cfg.get("n_trials", 100),
        "seed": cfg.get("seed", 0),
        "primitives": cfg.get("primitives", "all"),
        "check_nan_inf": cfg.get("check_nan_inf", False),
        "eps": EPS,
    }
    if "description" in cfg:
        new_cfg["description"] = cfg["description"].replace("forward-pass", "bounds")
    if "points" in cfg:
        new_cfg["points"] = cfg["points"]

    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "test.json").write_text(json.dumps(new_cfg, indent=2) + "\n")
    print(f"  created {dest_dir.relative_to(REPO_ROOT)}")


def main():
    for d in (UNIT_OUT, FUZZ_OUT, OPEN_FUZZ_OUT):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    rng_unit = np.random.default_rng(RNG_UNIT_SEED)
    rng_fuzz = np.random.default_rng(RNG_FUZZ_SEED)

    n_emitted = 0
    n_skipped = 0
    skip_reasons = []

    for ms1_dir in sorted(p for p in MS1_EVAL.iterdir() if p.is_dir()):
        cfg_path = ms1_dir / "test.json"
        if not cfg_path.exists():
            continue

        cfg = json.loads(cfg_path.read_text())
        network_rel = cfg["network"]
        graph_path = ms1_dir / network_rel
        graph = load(str(graph_path))

        prims = {eq.primitive.name for eq in graph.equations}
        unsupp = prims - IBP_SUPPORTED
        if unsupp:
            n_skipped += 1
            skip_reasons.append(f"  skip {ms1_dir.name}: unsupported primitives {sorted(unsupp)}")
            continue

        kinds = _decide_kinds(graph)

        try:
            unit_fix = _build_fixtures(ms1_dir, graph, kinds, N_UNIT_SAMPLES, rng_unit)
        except Exception as exc:
            n_skipped += 1
            skip_reasons.append(f"  skip {ms1_dir.name}: {type(exc).__name__}: {exc}")
            continue

        centers, boxes, ref_bounds, sample_outs, out_shapes = unit_fix
        _emit_test(
            UNIT_OUT, ms1_dir.name, ms1_dir, graph_path, kinds,
            centers, boxes, ref_bounds, sample_outs, out_shapes,
            with_reference=True,
        )

        fuzz_fix = _build_fixtures(ms1_dir, graph, kinds, N_FUZZ_SAMPLES, rng_fuzz)
        centers_f, boxes_f, ref_bounds_f, sample_outs_f, _ = fuzz_fix
        _emit_test(
            FUZZ_OUT, ms1_dir.name, ms1_dir, graph_path, kinds,
            centers_f, boxes_f, ref_bounds_f, sample_outs_f, out_shapes,
            with_reference=False,
        )

        n_emitted += 1

    print(f"\nEmitted {n_emitted} base test(s); skipped {n_skipped}.")
    for r in skip_reasons:
        print(r)

    # ---- Open hypothesis-fuzz tests (port milestone1 open/fuzz/eval configs) ----
    n_open_fuzz = 0
    for src in sorted(MS1_OPEN_FUZZ_EVAL.glob("*/test.json")):
        _port_open_fuzz_test(src, OPEN_FUZZ_OUT / src.parent.name)
        n_open_fuzz += 1
    print(f"\nEmitted {n_open_fuzz} open fuzz test(s).")


if __name__ == "__main__":
    main()
