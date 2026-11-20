#!/usr/bin/env python3
# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Launch only the isolated PM01 011 recipe; the coordinator chooses the GPU."""

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--sampling-index", type=Path)
    parser.add_argument("--robot-urdf", type=Path)
    parser.add_argument("--robot-spec", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluation-suites", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-compatibility-file", type=Path)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--pi-microbatch", type=int)
    parser.add_argument("--buffer-steps", type=int)
    parser.add_argument("--profile-actor-updates", type=int)
    parser.add_argument("--preflight-only", action="store_true", default=None)
    parser.add_argument("--max-vector-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--capture-graph", action=argparse.BooleanOptionalAction, default=None)
    cli = parser.parse_args()
    sys.path.insert(0, str(cli.source_root.resolve()))
    from holomotion.src.algo.supermimic.config import SuperMimicConfig
    from holomotion.src.algo.supermimic.training import execute
    values = dict(retrain_011=True, seed=13, amp="bf16", pi_batch=4096,
                  max_hours=0.0, max_vector_steps=0, max_updates=0,
                  skip_evaluation=True, policy_snapshot_minutes=0.0,
                  physics_backend="mujoco_warp", profile="paper", lr=2e-4)
    if cli.config:
        values.update(json.loads(cli.config.read_text()))
    for key, value in vars(cli).items():
        if key not in ("source_root", "config") and value is not None:
            values[key] = value
    values["utd"] = str(Fraction(2, 5) * int(values.get("num_envs", 4096)) / 4096)
    if not values.get("sampling_index") and values.get("manifest"):
        values["sampling_index"] = Path(values["manifest"]).with_name("sampling_index.json")
    args = SuperMimicConfig.from_mapping(values)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "launch_config.json").write_text(json.dumps({k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, indent=2) + "\n")
    execute(args)


if __name__ == "__main__":
    main()

