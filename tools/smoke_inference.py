#!/usr/bin/env python3
"""Run a deterministic, synthetic CPU inference check without a robot or motion data."""

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
from supermimic_pm01.policy_core import JOINT_NAMES, PolicyCore, ReferenceClip, RobotState


# Independent FP32 PyTorch evaluation of this same synthetic input using the
# 65k source policy. Allows small differences across CPU BLAS implementations.
EXPECTED_ACTION = np.array([
    -0.04331749, -0.22695425, 0.08588161, 0.19490677, 0.11190903,
    -0.00292689, -0.15353787, 0.07711274, -0.12412807, -0.10163239,
    0.05631833, 0.68489075, -0.72377712, -0.02188710, 0.71034610,
    0.04600778, -0.78576970, 0.21953264, -0.05767474, -0.06805931,
    0.19662161, -0.39957288, -0.33219764,
], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/actor_65000_fp32.onnx")
    parser.add_argument("--spec", type=Path, default=ROOT / "inference/robot_spec.json")
    args = parser.parse_args()

    with PolicyCore(args.model, args.spec, threads=1) as policy:
        q = policy.default_q.copy()
        zeros24 = np.zeros(24, dtype=np.float32)
        state = RobotState(q=q, qd=zeros24, root_quat_wxyz=np.array([1, 0, 0, 0], np.float32),
                           omega_body=np.zeros(3, dtype=np.float32))
        # Fixed root and joints: zero displacement, identity heading-relative
        # rotation (first two columns, row-major), zero velocities and q=default.
        one_goal = np.concatenate((np.zeros(3, np.float32), np.array([1, 0, 0, 1, 0, 0], np.float32),
                                   np.zeros(6, np.float32), q, zeros24))
        assert one_goal.shape == (63,)
        goal = np.tile(one_goal, 10)
        reference = ReferenceClip(q=np.repeat(q[None], 2, axis=0),
                                  goal=np.repeat(goal[None], 2, axis=0),
                                  fps=50.0, joint_names=JOINT_NAMES)

        first = policy.step(state, reference, 0)
        if first.action23.shape != (23,) or first.q_target.shape != (24,):
            raise RuntimeError("Unexpected policy output shape")
        if not np.isfinite(first.q_target).all():
            raise RuntimeError("Policy returned a non-finite target")
        max_error = float(np.max(np.abs(first.action23 - EXPECTED_ACTION)))
        if not np.allclose(first.action23, EXPECTED_ACTION, atol=2e-5, rtol=1e-5):
            raise RuntimeError(f"Synthetic Actor output differs from reference (max abs {max_error:.3g})")
        if not np.isclose(first.q_target[-1], q[-1], atol=1e-6):
            raise RuntimeError("Reference head target was not composed correctly")

        second = policy.step(state, reference, 1)
        if not np.isfinite(second.action23).all():
            raise RuntimeError("Second policy step returned a non-finite action")
        policy.reset()
        after_reset = policy.step(state, reference, 0)
        if not np.allclose(first.action23, after_reset.action23, atol=1e-6, rtol=0):
            raise RuntimeError("Reset did not reproduce the first output")

    print("CPU ONNX Runtime synthetic inference: PASS")
    print(f"Actor output [23], PD target [24]; max abs difference from PyTorch fixture: {max_error:.3g}")
    print("Two steps and reset passed. No physical robot, motion dataset, or simulator was used.")


if __name__ == "__main__":
    main()
