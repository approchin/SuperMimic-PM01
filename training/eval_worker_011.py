#!/usr/bin/env python3
# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Run one persistent evaluator on a coordinator-approved idle GPU."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--suites", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-envs", type=int, default=64)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    from holomotion.src.algo.supermimic.retrain_eval_worker import worker_main
    worker_main(run_root=args.run_root, suites_path=args.suites, device=args.device,
                max_envs=args.max_envs, once=args.once)


if __name__ == "__main__":
    main()

