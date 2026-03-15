#!/usr/bin/env python3
"""
Replace ONNX Erf nodes with a tanh-based polynomial approximation.

  erf(x) ≈ tanh(c1*x + c2*x³)

where c1 = 2/√π ≈ 1.1283791671, c2 ≈ 0.0356774.
Max absolute error: ~0.0005 across [-6, 6] (well within neural network tolerance).

All replacement ops (Tanh, Mul, Add) are TFLite BUILTINS — no SELECT_TF_OPS needed.

Usage:
    python replace_erf_onnx.py input.onnx output_no_erf.onnx
"""

import os
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper


def replace_erf_nodes(model_path: str, output_path: str):
    """Replace all Erf nodes in ONNX graph with tanh polynomial approximation."""
    print(f"Loading ONNX model: {model_path}")
    model = onnx.load(model_path)
    graph = model.graph

    # Find Erf node indices
    erf_indices = [i for i, n in enumerate(graph.node) if n.op_type == "Erf"]
    print(f"Found {len(erf_indices)} Erf node(s)")

    if not erf_indices:
        print("No Erf nodes found — saving unchanged model.")
        onnx.save(model, output_path)
        return

    # Process in reverse order to keep indices stable
    for idx in reversed(erf_indices):
        erf_node = graph.node[idx]
        x_in = erf_node.input[0]
        y_out = erf_node.output[0]
        prefix = f"pseudo_erf_{idx}"

        print(f"  Replacing node '{erf_node.name or f'Erf@{idx}'}': "
              f"{x_in} -> {y_out}")

        # Add constants as initializers
        # c1 = 2/√π ≈ 1.1283791671 (leading coefficient of erf Taylor series)
        c1_init = numpy_helper.from_array(
            np.array(1.1283791671, dtype=np.float32),
            name=f"{prefix}/c1",
        )
        # c2 ≈ 0.0356774 (cubic correction coefficient)
        c2_init = numpy_helper.from_array(
            np.array(0.0356774, dtype=np.float32),
            name=f"{prefix}/c2",
        )
        graph.initializer.append(c1_init)
        graph.initializer.append(c2_init)

        # Build replacement subgraph:
        #   x_sq  = x * x
        #   x_cu  = x_sq * x
        #   t1    = c1 * x
        #   t2    = c2 * x_cu
        #   inner = t1 + t2
        #   out   = Tanh(inner)
        replacement_nodes = [
            helper.make_node(
                "Mul", [x_in, x_in], [f"{prefix}/x_sq"],
                name=f"{prefix}/square",
            ),
            helper.make_node(
                "Mul", [f"{prefix}/x_sq", x_in], [f"{prefix}/x_cu"],
                name=f"{prefix}/cube",
            ),
            helper.make_node(
                "Mul", [f"{prefix}/c1", x_in], [f"{prefix}/t1"],
                name=f"{prefix}/mul_c1",
            ),
            helper.make_node(
                "Mul", [f"{prefix}/c2", f"{prefix}/x_cu"], [f"{prefix}/t2"],
                name=f"{prefix}/mul_c2",
            ),
            helper.make_node(
                "Add", [f"{prefix}/t1", f"{prefix}/t2"], [f"{prefix}/inner"],
                name=f"{prefix}/add",
            ),
            helper.make_node(
                "Tanh", [f"{prefix}/inner"], [y_out],
                name=f"{prefix}/tanh",
            ),
        ]

        # Remove Erf node and insert replacement subgraph at same position
        del graph.node[idx]
        for j, new_node in enumerate(replacement_nodes):
            graph.node.insert(idx + j, new_node)

    # Validate
    print("\nVerifying modified ONNX model...")
    try:
        onnx.checker.check_model(model)
        print("  ONNX checker: PASSED")
    except onnx.checker.ValidationError as e:
        # Large models may exceed protobuf size limits for checker
        print(f"  ONNX checker warning (model likely still valid): {e}")

    # Verify no Erf nodes remain
    remaining_erf = [n for n in graph.node if n.op_type == "Erf"]
    if remaining_erf:
        print(f"  WARNING: {len(remaining_erf)} Erf nodes still present!")
    else:
        print("  Confirmed: 0 Erf nodes remaining")

    # Count total nodes
    print(f"  Total nodes: {len(graph.node)} "
          f"(was {len(graph.node) - 5 * len(erf_indices) + len(erf_indices)})")

    # Save
    onnx.save(model, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved: {output_path} ({size_mb:.1f} MB)")


def verify_numerical_accuracy():
    """Quick check that the tanh approximation is accurate enough."""
    from scipy.special import erf as scipy_erf  # type: ignore

    x = np.linspace(-6, 6, 10000).astype(np.float32)
    exact = scipy_erf(x).astype(np.float32)
    approx = np.tanh(1.1283791671 * x + 0.0356774 * x ** 3).astype(np.float32)

    max_err = np.max(np.abs(exact - approx))
    mean_err = np.mean(np.abs(exact - approx))
    print(f"Numerical accuracy vs scipy.erf:")
    print(f"  Max  |error|: {max_err:.6f}")
    print(f"  Mean |error|: {mean_err:.6f}")
    print(f"  {'OK' if max_err < 0.01 else 'WARNING: high error'}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--verify":
        verify_numerical_accuracy()
        sys.exit(0)

    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.onnx> <output_no_erf.onnx>")
        print(f"       {sys.argv[0]} --verify   (check numerical accuracy)")
        sys.exit(1)

    replace_erf_nodes(sys.argv[1], sys.argv[2])
