"""
Compare state_dicts of two PyTorch models/checkpoints.
Usage:
    python compare_state_dicts.py --a path/to/native_rlinf.pt --b path/to/jax_converted.pt

You can also import and call compare_state_dicts() directly.
"""

import argparse
import torch
import numpy as np
from collections import OrderedDict


def load_state_dict(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path, device="cpu")
    
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "model_state_dict", "params"):
        if isinstance(ckpt, dict) and key in ckpt:
            print(f"  [{path}] Unwrapping checkpoint key: '{key}'")
            return ckpt[key]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Cannot extract state_dict from checkpoint at {path}")

def compare_state_dicts(sd_a: dict, sd_b: dict, label_a="A", label_b="B", rtol=1e-4, atol=1e-5):
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    only_in_a = sorted(keys_a - keys_b)
    only_in_b = sorted(keys_b - keys_a)
    common = sorted(keys_a & keys_b)

    print("\n" + "="*70)
    print(f"  {label_a}: {len(keys_a)} keys")
    print(f"  {label_b}: {len(keys_b)} keys")
    print(f"  Common: {len(common)} | Only in {label_a}: {len(only_in_a)} | Only in {label_b}: {len(only_in_b)}")
    print("="*70)

    # --- Keys only in A ---
    if only_in_a:
        print(f"\n[KEYS ONLY IN {label_a}] ({len(only_in_a)})")
        for k in only_in_a:
            print(f"  {k:60s}  shape={tuple(sd_a[k].shape) if hasattr(sd_a[k], 'shape') else '?'}")

    # --- Keys only in B ---
    if only_in_b:
        print(f"\n[KEYS ONLY IN {label_b}] ({len(only_in_b)})")
        for k in only_in_b:
            print(f"  {k:60s}  shape={tuple(sd_b[k].shape) if hasattr(sd_b[k], 'shape') else '?'}")

    # --- Common keys: shape and value comparison ---
    shape_mismatches = []
    dtype_mismatches = []
    value_mismatches = []
    close_matches = []

    print(f"\n[COMMON KEYS — DETAILED COMPARISON] ({len(common)})")
    print(f"  {'Key':<60}  {'Shape A':<20} {'Shape B':<20} {'Dtype A':<10} {'Dtype B':<10}  Status")
    print(f"  {'-'*60}  {'-'*20} {'-'*20} {'-'*10} {'-'*10}  ------")

    for k in common:
        ta = sd_a[k]
        tb = sd_b[k]

        shape_a = tuple(ta.shape) if hasattr(ta, 'shape') else None
        shape_b = tuple(tb.shape) if hasattr(tb, 'shape') else None
        dtype_a = str(ta.dtype) if hasattr(ta, 'dtype') else '?'
        dtype_b = str(tb.dtype) if hasattr(tb, 'dtype') else '?'

        if shape_a != shape_b:
            status = "SHAPE MISMATCH"
            shape_mismatches.append(k)
        elif dtype_a != dtype_b:
            status = "DTYPE MISMATCH"
            dtype_mismatches.append(k)
            # Still try value comparison after casting
            try:
                fa = ta.float().numpy()
                fb = tb.float().numpy()
                if np.allclose(fa, fb, rtol=rtol, atol=atol):
                    status += " (values close)"
                else:
                    diff = np.abs(fa - fb)
                    status += f" (max_diff={diff.max():.4e})"
                    value_mismatches.append((k, diff.max(), diff.mean()))
            except Exception:
                pass
        else:
            try:
                fa = ta.float().numpy()
                fb = tb.float().numpy()
                if np.allclose(fa, fb, rtol=rtol, atol=atol):
                    status = "OK"
                    close_matches.append(k)
                else:
                    diff = np.abs(fa - fb)

                    # Check if transposing fixes it
                    transpose_note = ""
                    if len(shape_a) == 2 and shape_a == shape_b[::-1]:
                        if np.allclose(fa, fb.T, rtol=rtol, atol=atol):
                            transpose_note = " *** MATCHES IF TRANSPOSED ***"

                    status = f"VALUE MISMATCH  max={diff.max():.4e}  mean={diff.mean():.4e}{transpose_note}"
                    value_mismatches.append((k, diff.max(), diff.mean()))
            except Exception as e:
                status = f"COMPARE ERROR: {e}"

        print(f"  {k:<60}  {str(shape_a):<20} {str(shape_b):<20} {dtype_a:<10} {dtype_b:<10}  {status}")

    # --- Summary ---
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Keys only in {label_a}:      {len(only_in_a)}")
    print(f"  Keys only in {label_b}:      {len(only_in_b)}")
    print(f"  Shape mismatches:         {len(shape_mismatches)}")
    print(f"  Dtype mismatches:         {len(dtype_mismatches)}")
    print(f"  Value mismatches:         {len(value_mismatches)}")
    print(f"  Clean matches:            {len(close_matches)}")

    if value_mismatches:
        print(f"\n  Top value mismatches by max diff:")
        for k, max_d, mean_d in sorted(value_mismatches, key=lambda x: -x[1])[:10]:
            print(f"    {k:<60}  max={max_d:.4e}  mean={mean_d:.4e}")

    if shape_mismatches:
        print(f"\n  Shape mismatches:")
        for k in shape_mismatches:
            print(f"    {k}")

    return {
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "value_mismatches": value_mismatches,
        "clean_matches": close_matches,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    a = '/mnt/data2/yi/logan/RLinf_models/RLinf-Pi05-LIBERO-SFT/model.safetensors'
    label_a = 'native_rlinf'
    b = '/mnt/data2/yi/logan/openpi/openpi-assets/checkpoints/pytorch_models/pi05_libero_pytorch/model.safetensors'
    label_b = 'jax_converted'
    print(f"\nLoading {a}...")
    sd_a = load_state_dict(a)
    print(f"Loading {b}...")
    sd_b = load_state_dict(b)

    compare_state_dicts(sd_a, sd_b, label_a=label_a, label_b=label_b,
                        rtol=args.rtol, atol=args.atol)