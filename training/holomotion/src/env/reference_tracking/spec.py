# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Robot/control dimensions shared by the reference method, no new learner."""
from dataclasses import dataclass
import json
from pathlib import Path

PM01_PAPER_TRACKED_BODIES=('LINK_BASE','LINK_HIP_ROLL_L','LINK_KNEE_PITCH_L','LINK_ANKLE_ROLL_L',
    'LINK_HIP_ROLL_R','LINK_KNEE_PITCH_R','LINK_ANKLE_ROLL_R','LINK_TORSO_YAW',
    'LINK_SHOULDER_ROLL_L','LINK_ELBOW_YAW_L','LINK_SHOULDER_ROLL_R','LINK_ELBOW_YAW_R','LINK_HEAD_YAW')


@dataclass(frozen=True)
class RobotSpec:
    kind: str
    joint_names: tuple[str, ...]
    policy_joint_names: tuple[str, ...]
    tracked_bodies: tuple[str, ...]
    root_body: str
    torso_body: str
    default_q: tuple[float, ...]
    action_scales: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    effort: tuple[float, ...]
    velocity: tuple[float, ...]
    armature: tuple[float, ...]
    mjcf: str | None = None
    future_stride: int = 1

    @property
    def joints(self): return len(self.joint_names)
    @property
    def actions(self): return len(self.policy_joint_names)
    @property
    def applied_actions(self): return self.joints
    @property
    def nq(self): return 7+self.joints
    @property
    def nv(self): return 6+self.joints
    @property
    def state_dim(self): return 13+2*self.joints
    @property
    def invariant_dim(self): return 13+2*self.joints
    @property
    def wm_delta_dim(self): return 12+2*self.joints
    @property
    def context_dim(self): return 4+self.joints
    @property
    def delay_state_dim(self): return 2*self.applied_actions+1 if self.kind == "pm01" else 0
    @property
    def proprio_dim(self): return 6+2*self.joints+self.applied_actions
    @property
    def goal_frame_dim(self): return 15+2*self.joints
    @property
    def critic_extra(self): return self.joints+len(self.tracked_bodies)*3+9+self.context_dim

    def compose_action(self, policy_action, reference_q, context=None):
        """Keep the actor23 path differentiable; head follows current reference."""
        import torch
        if policy_action.shape[-1] != self.actions:
            raise ValueError('Policy action dimension differs from robot contract')
        if self.kind == 'g1': return policy_action
        if self.joints != 24 or self.actions != 23 or self.joint_names[-1] != 'J23_HEAD_YAW':
            raise ValueError('Unsupported exogenous-head contract')
        default = torch.as_tensor(self.default_q, device=policy_action.device, dtype=policy_action.dtype)
        scale = torch.as_tensor(self.action_scales, device=policy_action.device, dtype=policy_action.dtype)
        offset = 0. if context is None else context[..., 1+self.joints-1:1+self.joints]
        head = (reference_q[..., -1:]-default[-1]-offset)/scale[-1]
        return torch.cat((policy_action, head), -1).clamp(-20., 20.)


def g1_spec():
    from .robot import G1_JOINT_NAMES, TRACKED_BODIES, default_joint_pos_list, action_scale, motor_of, _kp, _kd
    names = tuple(G1_JOINT_NAMES)
    return RobotSpec('g1', names, names, tuple(TRACKED_BODIES), 'pelvis', 'torso_link',
        tuple(default_joint_pos_list()), tuple(action_scale(n) for n in names),
        tuple(_kp(motor_of(n)[2]) for n in names), tuple(_kd(motor_of(n)[2]) for n in names),
        tuple(motor_of(n)[0] for n in names), tuple(motor_of(n)[1] for n in names),
        tuple(motor_of(n)[2] for n in names))


def load_robot_spec(path=None):
    if path is None: return g1_spec()
    d = json.loads(Path(path).read_text())
    if len(d['native_joint_names']) != 24 or len(d['policy_action_names']) != 23:
        raise ValueError('PM01 spec requires24 joints and23 learned actions')
    if (d['policy_action_names']!=d['native_joint_names'][:23] or d['native_joint_names'][-1]!='J23_HEAD_YAW'
        or tuple(d['tracking_body_names'])!=PM01_PAPER_TRACKED_BODIES or d['root_body']!='LINK_BASE'):
        raise ValueError('PM01 body-action/head/Any2Any point set cannot be silently changed')
    return RobotSpec('pm01', tuple(d['native_joint_names']), tuple(d['policy_action_names']),
        tuple(d['tracking_body_names']), d['root_body'], 'LINK_TORSO_YAW',
        *(tuple(d[k]) for k in ('default_joint_pos_rad','action_scale_rad','stiffness_nm_rad',
            'damping_nm_s_rad','effort_limit_nm','velocity_limit_rad_s','armature_kg_m2')),
        mjcf=d['mjcf'], future_stride=int(d.get('future_stride_frames', 5)))
