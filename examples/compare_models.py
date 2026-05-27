"""
JAX Model Comparison Diagnostic Script
======================================
Compares two JAX model parameter trees to identify mismatches in:
  - Key structure / missing keys
  - Shapes and dtypes
  - Numerical values (absolute and relative differences)

Usage:
    python compare_jax_models.py

Edit the LOAD section at the bottom to point at your two models.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any
import json
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_utils


# ─── Helpers ────────────────────────────────────────────────────────────────

def flatten_params(params: Any, prefix: str = "") -> dict:
    """
    Recursively flatten a nested dict/FrozenDict of arrays
    into a flat {dotted.key: array} dict.
    """
    flat = {}
    if hasattr(params, "_dict"):          # FrozenDict
        params = dict(params)
    if isinstance(params, dict):
        for k, v in params.items():
            full_key = f"{prefix}.{k}" if prefix else k
            flat.update(flatten_params(v, prefix=full_key))
    else:
        flat[prefix] = np.asarray(params)
    return flat


def compare_models(params_a: Any, params_b: Any,
                   name_a: str = "model_A", name_b: str = "model_B",
                   rtol: float = 1e-4, atol: float = 1e-5,
                   verbose_match: bool = False) -> dict:
    """
    Full comparison of two JAX parameter trees.

    Returns a results dict with keys:
        missing_in_b, extra_in_b, shape_mismatches,
        dtype_mismatches, value_mismatches, summary
    """
    flat_a = flatten_params(params_a)
    flat_b = flatten_params(params_b)

    keys_a = set(flat_a.keys())
    keys_b = set(flat_b.keys())

    missing_in_b = sorted(keys_a - keys_b)   # in A but not B
    extra_in_b   = sorted(keys_b - keys_a)   # in B but not A
    common_keys  = sorted(keys_a & keys_b)

    shape_mismatches  = []
    dtype_mismatches  = []
    value_mismatches  = []
    matched_keys      = []

    for key in common_keys:
        a = flat_a[key]
        b = flat_b[key]

        # Shape
        if a.shape != b.shape:
            shape_mismatches.append({
                "key": key,
                f"shape_{name_a}": str(a.shape),
                f"shape_{name_b}": str(b.shape),
            })
            continue  # can't compare values if shapes differ

        # Dtype
        dtype_ok = True
        if a.dtype != b.dtype:
            dtype_mismatches.append({
                "key": key,
                f"dtype_{name_a}": str(a.dtype),
                f"dtype_{name_b}": str(b.dtype),
            })
            dtype_ok = False

        # Values — cast to float32 for comparison if needed
        a_f = a.astype(np.float32)
        b_f = b.astype(np.float32)
        if not np.allclose(a_f, b_f, rtol=rtol, atol=atol):
            max_abs = float(np.max(np.abs(a_f - b_f)))
            mean_abs = float(np.mean(np.abs(a_f - b_f)))
            # relative diff guarded against zero denominator
            denom = np.abs(a_f) + 1e-8
            max_rel = float(np.max(np.abs(a_f - b_f) / denom))
            value_mismatches.append({
                "key": key,
                "shape": str(a.shape),
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "max_rel_diff": max_rel,
                "dtype_match": dtype_ok,
            })
        else:
            matched_keys.append(key)

    # Summary numbers
    total_params_a = sum(v.size for v in flat_a.values())
    total_params_b = sum(v.size for v in flat_b.values())

    results = {
        "summary": {
            "keys_in_A":          len(keys_a),
            "keys_in_B":          len(keys_b),
            "total_params_A":     total_params_a,
            "total_params_B":     total_params_b,
            "missing_in_B":       len(missing_in_b),
            "extra_in_B":         len(extra_in_b),
            "shape_mismatches":   len(shape_mismatches),
            "dtype_mismatches":   len(dtype_mismatches),
            "value_mismatches":   len(value_mismatches),
            "fully_matched_keys": len(matched_keys),
        },
        "missing_in_b":    missing_in_b,
        "extra_in_b":      extra_in_b,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "value_mismatches": sorted(value_mismatches,
                                   key=lambda x: -x["max_abs_diff"]),
    }

    return results


def print_report(results: dict, name_a: str = "model_A", name_b: str = "model_B"):
    s = results["summary"]
    sep = "─" * 60

    print(f"\n{'═' * 60}")
    print(f"  JAX Model Comparison: {name_a}  vs  {name_b}")
    print(f"{'═' * 60}")

    print(f"\n{'SUMMARY':}")
    print(sep)
    print(f"  Keys in {name_a}:            {s['keys_in_A']}")
    print(f"  Keys in {name_b}:            {s['keys_in_B']}")
    print(f"  Total params {name_a}:        {s['total_params_A']:,}")
    print(f"  Total params {name_b}:        {s['total_params_B']:,}")
    print(f"  Fully matching keys:        {s['fully_matched_keys']}")
    print(f"  Missing in {name_b}:          {s['missing_in_B']}")
    print(f"  Extra in {name_b}:            {s['extra_in_B']}")
    print(f"  Shape mismatches:           {s['shape_mismatches']}")
    print(f"  Dtype mismatches:           {s['dtype_mismatches']}")
    print(f"  Value mismatches:           {s['value_mismatches']}")

    if results["missing_in_b"]:
        print(f"\n⚠  KEYS IN {name_a} BUT NOT IN {name_b}:")
        print(sep)
        for k in results["missing_in_b"]:
            print(f"    {k}")

    if results["extra_in_b"]:
        print(f"\n⚠  KEYS IN {name_b} BUT NOT IN {name_a}:")
        print(sep)
        for k in results["extra_in_b"]:
            print(f"    {k}")

    if results["shape_mismatches"]:
        print(f"\n✗  SHAPE MISMATCHES:")
        print(sep)
        for m in results["shape_mismatches"]:
            print(f"  {m['key']}")
            for k, v in m.items():
                if k != "key":
                    print(f"    {k}: {v}")

    if results["dtype_mismatches"]:
        print(f"\n⚠  DTYPE MISMATCHES (may cause silent precision loss):")
        print(sep)
        for m in results["dtype_mismatches"]:
            print(f"  {m['key']}")
            for k, v in m.items():
                if k != "key":
                    print(f"    {k}: {v}")

    if results["value_mismatches"]:
        print(f"\n✗  VALUE MISMATCHES (top 20 by max abs diff):")
        print(sep)
        for m in results["value_mismatches"][:20]:
            print(f"  {m['key']}  shape={m['shape']}")
            print(f"    max_abs={m['max_abs_diff']:.6e}  "
                  f"mean_abs={m['mean_abs_diff']:.6e}  "
                  f"max_rel={m['max_rel_diff']:.6e}"
                  + ("  [DTYPE MISMATCH]" if not m["dtype_match"] else ""))
    else:
        print(f"\n✓  All common keys match within rtol/atol tolerances.")

    print(f"\n{'═' * 60}\n")


def spot_check_key(params_a, params_b, key: str,
                   name_a: str = "model_A", name_b: str = "model_B"):
    """
    Deep-dive a single parameter key.
    Useful for understanding *how* weights differ (transposed? scaled? zeroed?).
    """
    flat_a = flatten_params(params_a)
    flat_b = flatten_params(params_b)

    if key not in flat_a:
        print(f"Key '{key}' not found in {name_a}")
        return
    if key not in flat_b:
        print(f"Key '{key}' not found in {name_b}")
        return

    a = flat_a[key].astype(np.float32)
    b = flat_b[key].astype(np.float32)

    print(f"\nSpot-check: {key}")
    print(f"  Shape: {a.shape}  (both)")
    print(f"  Dtype {name_a}: {flat_a[key].dtype},  {name_b}: {flat_b[key].dtype}")
    print(f"  {name_a} stats: min={a.min():.4f}  max={a.max():.4f}  mean={a.mean():.4f}  std={a.std():.4f}")
    print(f"  {name_b} stats: min={b.min():.4f}  max={b.max():.4f}  mean={b.mean():.4f}  std={b.std():.4f}")
    diff = a - b
    print(f"  Diff stats:    min={diff.min():.4f}  max={diff.max():.4f}  mean={diff.mean():.4f}  std={diff.std():.4f}")

    # Check if b ≈ a transposed (common conversion bug for 2-D weights)
    if a.ndim == 2 and a.shape == b.T.shape:
        if np.allclose(a, b.T, atol=1e-4):
            print(f"  ⚠  TRANSPOSITION MATCH: b.T ≈ a  — likely a weight transpose bug!")
        else:
            print(f"  (not a simple transpose)")

    # Check if b ≈ a * scalar
    ratio = b / (a + 1e-8)
    ratio_std = float(ratio.std())
    if ratio_std < 1e-3:
        print(f"  ⚠  SCALE MATCH: b ≈ a × {ratio.mean():.4f}  — likely a scale/normalisation bug!")

    # First few values for eyeballing
    print(f"  First 5 flat values  {name_a}: {a.flat[:5]}")
    print(f"  First 5 flat values  {name_b}: {b.flat[:5]}")


# ─── LOAD YOUR MODELS HERE ───────────────────────────────────────────────────
#
# Replace the two blocks below with however you load your params.
# Both should produce a nested dict (or FrozenDict) of jax/numpy arrays.
#
# Examples:
#
#   import pickle
#   with open("model_original.pkl", "rb") as f:
#       params_original = pickle.load(f)
#
#   from flax.training import checkpoints
#   params_original = checkpoints.restore_checkpoint("ckpt_original", target=None)
#
#   import orbax.checkpoint as ocp
#   params_original = ocp.PyTreeCheckpointer().restore("ckpt_original")
#
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CKPT_ORIGINAL   = "/mnt/data2/yi/logan/openpi/openpi-assets/checkpoints/pi05_libero/params"
    CKPT_ROUNDTRIP  = "/mnt/data2/yi/logan/openpi/openpi-assets/checkpoints/pi05_libero_converted_jax/params" 
    
    def restore_to_cpu(ckpt_path: str) -> dict:
        checkpointer = ocp.PyTreeCheckpointer()
        cpu = jax.devices("cpu")[0]
        sharding = jax.sharding.SingleDeviceSharding(cpu)

        metadata = checkpointer.metadata(ckpt_path)

        # Build a parallel tree of ArrayRestoreArgs with explicit sharding
        def make_arg(m):
            return ocp.ArrayRestoreArgs(
                restore_type=jax.Array,
                sharding=sharding,
            )

        restore_args = jax.tree_util.tree_map(make_arg, metadata)

        return checkpointer.restore(
            ckpt_path,
            item=None,
            restore_args=restore_args,
        )

    params_original  = restore_to_cpu(CKPT_ORIGINAL)
    params_roundtrip = restore_to_cpu(CKPT_ROUNDTRIP)

    results = compare_models(
        params_original,
        params_roundtrip,
        name_a="original",
        name_b="roundtrip",
        rtol=1e-4,
        atol=1e-5,
    )

    print_report(results, name_a="original", name_b="roundtrip")

    # Optional: drill into the worst offender
    if results["value_mismatches"]:
        worst_key = results["value_mismatches"][0]["key"]
        spot_check_key(params_original, params_roundtrip, worst_key,
                       name_a="original", name_b="roundtrip")

    # Optional: dump full report to JSON for further analysis
    with open("examples/comparison_report.json", "w") as f:
        json.dump(results, f, indent=2, default=str)