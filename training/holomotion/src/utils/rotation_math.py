# SPDX-License-Identifier: Apache-2.0
"""Rotation algebra used by the public PM01 data loader.

This release-specific implementation provides only the eleven interfaces used
by h5_dataloader and reference_fk. It replaces the separately licensed utility
file from the historical training environment. Quaternions use WXYZ unless an
interface explicitly names XYZW. Inputs to rotation functions must be nonzero
quaternions representing rotations; matrix inputs must be proper rotations.
"""

import torch


def xyzw_to_wxyz(q):
    return q[..., (3, 0, 1, 2)]


def wxyz_to_xyzw(q):
    return q[..., (1, 2, 3, 0)]


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def standardize_quaternion(q):
    return torch.where(q[..., :1] < 0, q.neg(), q)


def quat_mul(left, right):
    """Hamilton product, including broadcast batch dimensions."""
    left, right = torch.broadcast_tensors(left, right)
    a, b = left[..., 1:], right[..., 1:]
    scalar = left[..., :1] * right[..., :1] - (a * b).sum(-1, keepdim=True)
    vector = left[..., :1] * b + right[..., :1] * a + torch.cross(a, b, dim=-1)
    return torch.cat((scalar, vector), dim=-1)


def matrix_from_quat(q):
    """Convert nonzero WXYZ quaternions to active rotation matrices."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def quat_apply(q, vector):
    """Rotate vectors by nonzero WXYZ quaternions."""
    return (matrix_from_quat(q) @ vector.unsqueeze(-1)).squeeze(-1)


def quat_from_matrix(matrix):
    """Convert SO(3) matrices to unit WXYZ quaternions, choosing w >= 0.

    Choose the largest of the four squared quaternion components before
    division, so rotations close to pi do not require division by a small w.
    """
    a, b, c = matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2]
    squared = torch.stack((1 + a + b + c, 1 + a - b - c,
                           1 - a + b - c, 1 - a - b + c), dim=-1)
    skew_x = matrix[..., 2, 1] - matrix[..., 1, 2]
    skew_y = matrix[..., 0, 2] - matrix[..., 2, 0]
    skew_z = matrix[..., 1, 0] - matrix[..., 0, 1]
    sym_xy = matrix[..., 0, 1] + matrix[..., 1, 0]
    sym_xz = matrix[..., 0, 2] + matrix[..., 2, 0]
    sym_yz = matrix[..., 1, 2] + matrix[..., 2, 1]
    rows = torch.stack(
        (
            torch.stack((squared[..., 0], skew_x, skew_y, skew_z), dim=-1),
            torch.stack((skew_x, squared[..., 1], sym_xy, sym_xz), dim=-1),
            torch.stack((skew_y, sym_xy, squared[..., 2], sym_yz), dim=-1),
            torch.stack((skew_z, sym_xz, sym_yz, squared[..., 3]), dim=-1),
        ),
        dim=-2,
    )
    candidates = rows / (2 * squared.clamp_min(1e-12).sqrt()).unsqueeze(-1)
    choice = squared.argmax(dim=-1)
    gather_at = choice[..., None, None].expand(choice.shape + (1, 4))
    q = candidates.gather(-2, gather_at).squeeze(-2)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    return standardize_quaternion(q)


def axis_angle_to_matrix(angles, axes):
    """Rotate each joint by its angle about its nonzero axis (N,3)."""
    lengths = torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
    if bool(torch.any(lengths <= 0)):
        raise ValueError("Rotation axes must be nonzero")
    directions = axes / lengths
    half = angles.unsqueeze(-1) * 0.5
    q = torch.cat((half.cos(), half.sin() * directions), dim=-1)
    return matrix_from_quat(q)


def axis_angle_from_quat(q, w_last=True):
    """Principal rotation vector with angle in [0, pi]."""
    if w_last:
        q = xyzw_to_wxyz(q)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    q = standardize_quaternion(q)
    v = q[..., 1:]
    sin_half = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, q[..., :1])
    # sin(theta/2)/(theta/2) has a finite limit of one at zero.
    scale = 2 / torch.sinc(angle / (2 * torch.pi))
    return v * scale


def grad_t(values, dt):
    """Time derivative along dimension 1; central interior, one-sided ends."""
    if dt <= 0:
        raise ValueError("Time spacing must be positive")
    if values.shape[1] < 2:
        return torch.zeros_like(values)
    return torch.gradient(values, spacing=(dt,), dim=(1,), edge_order=1)[0]
