# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Load the shared physical task for training and saved-policy evaluation."""

import torch

from holomotion.src.env.reference_tracking.kinematics import G1Kinematics
from holomotion.src.env.reference_tracking.mdp import MDP
from holomotion.src.env.reference_tracking.model import build_model
from holomotion.src.env.reference_tracking.motion import MotionLib
from holomotion.src.env.reference_tracking.robot import default_joint_pos_list


def load_task(*, robot_urdf, manifest, output, device, profile, robot_spec=None):
    from .spec import load_robot_spec
    spec=load_robot_spec(robot_spec)
    if spec.kind=='pm01' and profile!='paper':
        raise ValueError('PM01 candidate retains self-collisions; vibed is not this task')
    model = build_model(
        robot_urdf, output / ("pm01.xml" if spec.kind=='pm01' else "g1.xml"), self_collisions=profile == "paper",robot_spec=spec
    )
    kinematics = G1Kinematics(device, model,robot_spec=spec)
    motion = MotionLib.from_hdf5(manifest, kinematics, robot_urdf, device)
    mdp = MDP(
        kinematics,
        motion,
        torch.tensor(spec.default_q, device=device),
        reference_orientation=profile == "vibed",
    )
    return model, motion, mdp
