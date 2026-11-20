"""PM01 SuperMimic 65k, 50 Hz, one policy sequence per PolicyCore.

The caller supplies calibrated model-space joints and LINK_BASE orientation/gyro.
Clips carry training MotionLib goals computed ahead of time. This module does
no networking, motor I/O, timing, or simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import operator
from pathlib import Path

import numpy as np


JOINT_NAMES = (
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L", "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J06_HIP_PITCH_R", "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_WAIST_YAW", "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L", "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R", "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R", "J23_HEAD_YAW",
)
IO_SHAPES = {"obs_hist": (1, 10, 78), "goal": (1, 630), "action23": (1, 23)}


def _array(value, shape, name):
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape or not np.isfinite(arr).all():
        raise ValueError(f"{name} requires finite float values with shape {shape}, got {arr.shape}")
    return arr


@dataclass(frozen=True)
class RobotState:
    """q [rad], qd [rad/s] in JOINT_NAMES order. Root quaternion is world-from-LINK_BASE."""

    q: np.ndarray
    qd: np.ndarray
    root_quat_wxyz: np.ndarray
    omega_body: np.ndarray  # Angular velocity of LINK_BASE expressed in LINK_BASE, rad/s.


@dataclass(frozen=True)
class ReferenceClip:
    q: np.ndarray
    goal: np.ndarray
    fps: float
    joint_names: tuple[str, ...]
    qd: np.ndarray | None = None
    root_quat_wxyz: np.ndarray | None = None
    root_p: np.ndarray | None = None

    @property
    def frames(self):
        return len(self.q)

    def __len__(self):
        return self.frames

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as file:
            fps = float(file["fps"].item())
            joint_names = tuple(file["joint_names"].tolist())
            if fps != 50.0 or joint_names != JOINT_NAMES:
                raise ValueError("Expected a 50-Hz reference in native PM01 model joint order")
            q = file["q"]
            frames = len(q)
            if frames < 2:
                raise ValueError("A reference needs at least two control frames")
            checked = {"q": _array(q, (frames, 24), "reference q"),
                       "goal": _array(file["goal"], (frames, 630), "reference goal")}
            for name, width in (("qd", 24), ("root_quat_wxyz", 4), ("root_p", 3)):
                if name in file:
                    checked[name] = _array(file[name], (frames, width), name)
        return cls(fps=fps, joint_names=joint_names, **checked)


@dataclass(frozen=True)
class PolicyOutput:
    action23: np.ndarray  # Deterministic Actor output, before clipping.
    action24: np.ndarray  # Clipped command including exogenous head, used in next observation.
    q_target: np.ndarray  # Native 24-joint PD target in rad.


def projected_gravity(root_quat_wxyz):
    """R_world_from_base.T @ [0, 0, -1]. No accelerometer magnitude or heading alignment."""
    quat = _array(root_quat_wxyz, (4,), "root_quat_wxyz")
    norm = np.linalg.norm(quat)
    if norm < 1e-6 or not np.isfinite(norm):
        raise ValueError("LINK_BASE orientation requires a nonzero quaternion")
    w, x, y, z = quat / norm
    two = np.float32(2)
    return np.array([two*(w*y-x*z), -two*(y*z+w*x), two*(x*x+y*y)-1], dtype=np.float32)


class _OrtActor:
    def __init__(self, path, threads):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if [x.name for x in inputs] != ["obs_hist", "goal"] or [x.name for x in outputs] != ["action23"]:
            raise ValueError("Unexpected inputs or outputs for the 65k Actor")
        for item in inputs + outputs:
            if tuple(item.shape) != IO_SHAPES[item.name] or item.type != "tensor(float)":
                raise ValueError(f"Invalid ONNX tensor {item.name}: {item.type} {item.shape}")

    def run(self, history, goal):
        return self.session.run(["action23"], {"obs_hist": history, "goal": goal})[0]

    def close(self):
        self.session = None


class PolicyCore:
    """Call reset() once per clip, then step() once for each issued 20 ms command.

    Use one thread at a time. On a rejected/modified command, stop this sequence.
    Restart from a fresh reset or restore the actual issued history for a replay.
    """
    def __init__(self, model_path, spec_path, calibration_rad=None, threads=1):
        cfg = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        self.joint_names = tuple(cfg["native_joint_names"])
        if (self.joint_names != JOINT_NAMES or tuple(cfg["policy_action_names"]) != JOINT_NAMES[:23]
                or cfg["root_body"] != "LINK_BASE" or cfg["raw_action_clip"] != 20
                or cfg["control_dt_s"] != 0.02 or cfg["future_stride_frames"] != 5):
            raise ValueError("Robot specification differs from the 65k training contract")
        self.default_q = _array(cfg["default_joint_pos_rad"], (24,), "default_joint_pos_rad")
        self.scale = _array(cfg["action_scale_rad"], (24,), "action_scale_rad")
        self.kp = _array(cfg["stiffness_nm_rad"], (24,), "stiffness_nm_rad")
        self.kd = _array(cfg["damping_nm_s_rad"], (24,), "damping_nm_s_rad")
        self.effort = _array(cfg["effort_limit_nm"], (24,), "effort_limit_nm")
        if not all((v > 0).all() for v in (self.scale, self.kp, self.kd, self.effort)):
            raise ValueError("Action scales, gains and effort limits must be positive")
        self.calibration = (np.zeros(24, np.float32) if calibration_rad is None
                            else _array(calibration_rad, (24,), "calibration_rad").copy())
        path = Path(model_path)
        if path.suffix.lower() == ".onnx":
            if operator.index(threads) < 1:
                raise ValueError("CPU inference thread count must be positive")
            self._actor = _OrtActor(path, threads)
        else:
            raise ValueError("Expected the exported .onnx model")
        self.reset()

    def reset(self):
        self._history = None
        self._last_issued = np.zeros(24, dtype=np.float32)

    def commit_issued_target(self, q_target):
        """Commit the actual sent target after an explicit deployment transform.

        step() retains its original default behavior. Call this only when an
        issued target differs, before observing the next physical state.
        """
        target = _array(q_target, (24,), "actual issued PD target")
        action = (target-self.default_q-self.calibration)/self.scale
        _array(action, (24,), "actual issued action")
        if (np.abs(action) > 20.0).any():
            raise ValueError("Actual issued target exceeds the source raw_action_clip=20 envelope")
        self._last_issued = action.copy()
        return action.copy()

    def restore_history(self, history_through_previous, previous_issued_action):
        """Resume an independently recorded snapshot directly before observing current state."""
        history = _array(history_through_previous, (10, 78), "prior observation history")
        previous = _array(previous_issued_action, (24,), "previous issued action")
        self._history = history.copy()
        self._last_issued = previous.copy()

    def observe(self, state: RobotState):
        q = _array(state.q, (24,), "q")
        qd = _array(state.qd, (24,), "qd")
        omega = _array(state.omega_body, (3,), "omega_body")
        gravity = projected_gravity(state.root_quat_wxyz)
        return np.concatenate((q - self.default_q - self.calibration, qd, omega, gravity, self._last_issued))

    def infer(self, history, goal):
        history = _array(history, (10, 78), "history")
        goal = _array(goal, (630,), "goal")
        action = self._actor.run(history[None], goal[None])
        return _array(action, (1, 23), "Actor output")[0].copy()

    def to_target(self, action23, reference_q):
        action23 = _array(action23, (23,), "Actor output")
        head_q = _array(reference_q, (24,), "reference q")[-1:]
        head = (head_q - self.default_q[-1:] - self.calibration[-1:]) / self.scale[-1:]
        action24 = np.clip(np.concatenate((action23, head)), -20.0, 20.0)
        target = self.default_q + self.calibration + self.scale * action24
        _array(target, (24,), "PD target")
        return PolicyOutput(action23, action24, target)

    def step(self, state: RobotState, reference: ReferenceClip, frame: int):
        index = operator.index(frame)
        if not 0 <= index < reference.frames:
            raise IndexError(f"Reference frame {index} outside this {reference.frames}-frame clip")
        if reference.joint_names != self.joint_names or reference.fps != 50:
            raise ValueError("Incompatible joint order or control rate in reference")
        obs = self.observe(state)
        if self._history is None:
            history = np.repeat(obs[None], 10, axis=0)
        else:
            history = np.concatenate((self._history[1:], obs[None]), axis=0)
        output = self.to_target(self.infer(history, reference.goal[index]), reference.q[index])
        self._history = history
        self._last_issued = output.action24.copy()
        return output

    def close(self):
        self._actor.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
