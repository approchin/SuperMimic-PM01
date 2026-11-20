#!/usr/bin/env python3
"""Verify the published 65k model files and, optionally, every checkpoint tensor."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
EXPECTED_FILES = {"actor_0000065000.pt", "actor_65000_fp32.onnx"}


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files():
    manifest = json.loads((MODELS / "SHA256SUMS.json").read_text(encoding="utf-8"))
    if manifest.get("algorithm") != "sha256" or set(manifest["files"]) != EXPECTED_FILES:
        raise ValueError("Invalid model manifest")
    for name, info in manifest["files"].items():
        path = MODELS / name
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} is missing from models/. Place the published model attachment there."
            )
        if path.stat().st_size != info["bytes"] or digest_file(path) != info["sha256"]:
            raise ValueError(f"Size or SHA-256 mismatch for {name}")
        print(f"OK {name}: {info['sha256']}")


def verify_tensors():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("--tensors requires PyTorch; file verification does not") from exc
    reference = json.loads((MODELS / "actor_tensor_sha256.json").read_text(encoding="utf-8"))["tensors"]
    bundle = torch.load(MODELS / "actor_0000065000.pt", map_location="cpu", weights_only=True)
    tensors = bundle["pi"]
    if set(tensors) != set(reference):
        raise ValueError("Checkpoint tensor names do not match the reference manifest")
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Checkpoint value is not a tensor: {name}")
        actual = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(
                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
        if actual != reference[name]:
            raise ValueError(f"Tensor content mismatch: {name}")
    print(f"OK {len(tensors)} tensors: names, shapes, dtypes and contents")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensors", action="store_true", help="also verify tensor content (requires PyTorch)")
    args = parser.parse_args()
    try:
        verify_files()
        if args.tensors:
            verify_tensors()
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
