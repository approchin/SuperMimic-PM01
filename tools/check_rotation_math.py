# SPDX-License-Identifier: Apache-2.0
"""CPU checks of public data-loader rotation helpers against SciPy."""

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "training/holomotion/src/utils/rotation_math.py"
    spec = importlib.util.spec_from_file_location("pm01_rotation_math", path)
    math = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math)
    rng = np.random.default_rng(13)
    random = Rotation.random(1024, random_state=rng)
    boundaries = Rotation.from_rotvec(np.array([
        [0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0], [0, 0, np.pi],
        [1e-12, 0, 0], [np.pi - 1e-9, 0, 0],
    ]))
    rotations = Rotation.concatenate([random, boundaries])
    xyzw = torch.from_numpy(rotations.as_quat())
    q = math.xyzw_to_wxyz(xyzw)
    matrices = math.matrix_from_quat(q)
    errors = {}

    def compare(label, actual, expected, tolerance=1e-10):
        actual = np.asarray(actual.detach().cpu() if isinstance(actual, torch.Tensor) else actual)
        expected = np.asarray(expected)
        error = float(np.max(np.abs(actual - expected)))
        if not np.isfinite(error) or error > tolerance:
            raise AssertionError(f"{label}: {error} exceeds {tolerance}")
        errors[label] = error

    compare("quaternion_order_roundtrip", math.wxyz_to_xyzw(q), xyzw)
    compare("rotation_matrices", matrices, rotations.as_matrix())
    compare("nonunit_quaternions", math.matrix_from_quat(3 * q), rotations.as_matrix())
    recovered = math.quat_from_matrix(matrices)
    compare("matrix_roundtrip", math.matrix_from_quat(recovered), rotations.as_matrix())
    compare("negative_quaternions", math.matrix_from_quat(-q), rotations.as_matrix())
    compare("conjugate_inverse", math.matrix_from_quat(math.quat_conjugate(q)),
            rotations.inv().as_matrix())
    points = rng.normal(size=(len(q), 3))
    compare("rotated_vectors", math.quat_apply(q, torch.from_numpy(points)),
            rotations.apply(points))
    second = Rotation.random(len(q), random_state=rng)
    right = math.xyzw_to_wxyz(torch.from_numpy(second.as_quat()))
    compare("hamilton_product", math.matrix_from_quat(math.quat_mul(q, right)),
            (rotations * second).as_matrix())
    principal = math.axis_angle_from_quat(q, w_last=False)
    compare("rotation_vectors", Rotation.from_rotvec(principal.numpy()).as_matrix(),
            rotations.as_matrix())
    compare("xyzw_rotation_vectors", math.axis_angle_from_quat(xyzw), principal)
    angles = torch.from_numpy(rng.uniform(-np.pi, np.pi, (2, 5, 7)))
    axes = torch.from_numpy(rng.normal(size=(7, 3)))
    rotvec = angles[..., None] * (axes / axes.norm(dim=-1, keepdim=True))
    expected = Rotation.from_rotvec(rotvec.numpy().reshape(-1, 3)).as_matrix()
    compare("joint_axes", math.axis_angle_to_matrix(angles, axes).reshape(-1, 3, 3), expected)
    q32 = q.float()
    matrices32 = math.matrix_from_quat(q32)
    compare("float32_matrices", matrices32, rotations.as_matrix(), tolerance=2e-6)
    compare("float32_matrix_roundtrip",
            math.matrix_from_quat(math.quat_from_matrix(matrices32)),
            rotations.as_matrix(), tolerance=2e-6)
    rotvec32 = math.axis_angle_from_quat(q32, w_last=False)
    compare("float32_rotation_vectors",
            Rotation.from_rotvec(rotvec32.numpy()).as_matrix(),
            rotations.as_matrix(), tolerance=2e-6)
    samples = torch.from_numpy(rng.normal(size=(2, 9, 4)))
    compare("time_derivative", math.grad_t(samples, 0.02),
            np.gradient(samples.numpy(), 0.02, axis=1))
    compare("single_frame_derivative", math.grad_t(samples[:, :1], 0.02),
            np.zeros((2, 1, 4)))
    for operation in [lambda: math.grad_t(samples, 0),
                      lambda: math.axis_angle_to_matrix(angles, torch.zeros_like(axes))]:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid spacing or axis was accepted")
    print(json.dumps({"status": "passed", "device": "cpu", "rotations": len(q),
                      "maximum_absolute_errors": errors}, indent=2))


if __name__ == "__main__":
    main()
