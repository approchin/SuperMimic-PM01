# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Validated training options shared by CLI and framework configuration."""

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

from holomotion.src.algo.supermimic.schedule import rate


@dataclass(frozen=True)
class SuperMimicConfig:
    """Reference recipe; logical batches remain distinct from microbatches."""

    manifest: Path
    robot_urdf: Path
    output: Path
    robot_spec: Path | None = None
    validation_manifest: Path | None = None
    evaluation_max_envs: int = 64
    resume: Path | None = None
    resume_compatibility_file: Path | None = None
    device: str = "cuda:0"
    profile: str = "paper"
    num_envs: int = 4096
    buffer_steps: int = 1000
    max_hours: float = 2.0
    max_vector_steps: int = 800064
    max_updates: int = 320000
    utd: str = "0.4"
    horizon: int = 32
    goal_frames: int = 10
    hist: int = 10
    wm_window: int = 8
    wm_batches: int = 10
    wm_batch: int = 2048
    pi_batch: int = 4096
    pi_microbatch: int = 512
    lr: float = 0.0002
    gamma: float = 0.99
    lam: float = 0.95
    action_noise: float = 0.1
    critic_epochs: int = 5
    critic_minibatches: int = 4
    no_critic: bool = False
    warmup_steps: int = 64
    world_only_updates: int = 500
    lr_decay_updates: int = 250000
    policy_snapshot_minutes: float = 0.0
    checkpoint_minutes: float = 15.0
    evaluate_minutes: float = 30.0
    log_steps: int = 50
    seed: int = 0
    amp: str = "bf16"
    compile: bool = True
    capture_graph: bool = True
    release_phase_cache: bool = True
    preflight_only: bool = False
    skip_evaluation: bool = False
    physics_backend: str = "mujoco_warp"
    retrain_011: bool = False
    sampling_index: Path | None = None
    evaluation_suites: Path | None = None
    resume_interval_steps: int = 1000
    actor_snapshot_start: int = 50000
    actor_snapshot_interval: int = 1000
    full_evaluation_interval: int = 10000
    scalar_flush_records: int = 256
    divergence_window: int = 100
    divergence_windows: int = 3
    divergence_factor: float = 5.0
    quick_check_timeout_s: float = 180.0
    profile_actor_updates: int = 0

    @classmethod
    def from_mapping(cls, values: Mapping):
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"Unknown SuperMimic options: {sorted(unknown)}")
        converted = dict(values)
        for key in (
            "resume",
            "resume_compatibility_file",
            "manifest",
            "robot_urdf",
            "robot_spec",
            "validation_manifest",
            "output",
            "sampling_index",
            "evaluation_suites",
        ):
            if converted.get(key) is not None:
                converted[key] = Path(converted[key]).expanduser().resolve()
        result = cls(**converted)
        result.validate()
        return result

    def validate(self):
        for key in (
            "compile",
            "capture_graph",
            "release_phase_cache",
            "no_critic",
            "preflight_only",
            "skip_evaluation",
            "retrain_011",
        ):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a boolean")
        if type(self.seed) is not int:
            raise ValueError("Seed must be an integer")
        positive = (
            "num_envs",
            "evaluation_max_envs",
            "buffer_steps",
            "horizon",
            "goal_frames",
            "hist",
            "wm_window",
            "wm_batches",
            "wm_batch",
            "pi_batch",
            "pi_microbatch",
            "critic_epochs",
            "critic_minibatches",
            "lr_decay_updates",
            "log_steps",
            "resume_interval_steps",
            "actor_snapshot_start",
            "actor_snapshot_interval",
            "full_evaluation_interval",
            "scalar_flush_records",
            "divergence_window",
            "divergence_windows",
        )
        for key in positive:
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("warmup_steps", "world_only_updates", "profile_actor_updates", "max_vector_steps", "max_updates"):
            value = getattr(self, key)
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in (
            "lr",
            "checkpoint_minutes",
            "evaluate_minutes",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not math.isfinite(self.max_hours) or self.max_hours < 0:
            raise ValueError("max_hours must be nonnegative; zero disables the time limit")
        if not math.isfinite(self.divergence_factor) or self.divergence_factor <= 1:
            raise ValueError("divergence_factor must exceed one")
        if not math.isfinite(self.quick_check_timeout_s) or self.quick_check_timeout_s <= 0:
            raise ValueError("quick_check_timeout_s must be positive")
        if self.retrain_011:
            from fractions import Fraction
            if self.seed != 13 or self.pi_batch != 4096 or self.no_critic or self.amp != "bf16":
                raise ValueError("011 requires seed13, effective pi_batch4096, Critic and BF16")
            if rate(self.utd) != Fraction(2, 5) * self.num_envs / 4096:
                raise ValueError("011 UTD must preserve the approved per-transition ratio: 0.4*N/4096")
            if self.physics_backend != "mujoco_warp" or self.profile != "paper" or self.robot_spec is None:
                raise ValueError("011 uses the original PM01 paper/native-Warp contract")
        if (
            not math.isfinite(self.policy_snapshot_minutes)
            or self.policy_snapshot_minutes < 0
        ):
            raise ValueError(
                "Policy snapshot minutes must be finite and nonnegative"
            )
        if self.resume is not None and self.policy_snapshot_minutes > 0:
            raise ValueError(
                "Policy progress snapshots cannot be combined with resume"
            )
        if not math.isfinite(self.action_noise) or self.action_noise < 0:
            raise ValueError("Action noise must be finite and nonnegative")
        if not 0 <= self.gamma < 1 or not 0 <= self.lam <= 1:
            raise ValueError("Invalid discount or lambda")
        if self.profile not in ("paper", "vibed") or self.amp not in (
            "none",
            "bf16",
        ):
            raise ValueError("Unknown observation profile or precision")
        if self.physics_backend not in ("mujoco_warp", "isaac_physx"):
            raise ValueError("Unknown physics backend")
        if rate(self.utd) <= 0:
            raise ValueError("Update-to-data rate must be positive")
        first_fit_step = self.warmup_steps + math.ceil(1 / rate(self.utd))
        if not self.preflight_only and first_fit_step < self.wm_window:
            raise ValueError(
                "Warmup must populate a complete world-model window before its first update"
            )
        if self.buffer_steps < max(
            self.wm_window, self.hist, self.warmup_steps
        ):
            raise ValueError(
                "Replay must retain model windows, policy history and warmup"
            )
        if (self.pi_batch * self.horizon) % self.critic_minibatches:
            raise ValueError(
                "Critic minibatches must cover every imagined sample"
            )
        if not self.manifest.is_file() or not self.robot_urdf.is_file():
            raise ValueError("Manifest and original robot URDF must exist")
        if self.validation_manifest is not None:
            if not self.validation_manifest.is_file():
                raise ValueError("Validation manifest must exist")
            if self.validation_manifest == self.manifest:
                raise ValueError("Validation must use a separate manifest")
            if self.physics_backend != "mujoco_warp":
                raise ValueError("Separate validation currently requires MuJoCo Warp")
        if self.robot_spec is not None:
            if not self.robot_spec.is_file() or self.physics_backend!='mujoco_warp' or self.profile!='paper':
                raise ValueError('PM01 candidate requires an actual spec and the native Warp paper/self-collision path')
        if self.output.exists() and any(self.output.iterdir()) and self.resume is None and not self.retrain_011:
            raise ValueError("Choose an empty training output directory")
        if self.retrain_011 and self.resume is None:
            prior = self.output / "logs/scalars.jsonl"
            if prior.exists() and prior.stat().st_size:
                raise ValueError("011 already contains collected training logs; use its latest_resume, never silently cold-start over it")
        if self.resume is not None:
            if not self.resume.is_file() or self.preflight_only:
                raise ValueError(
                    "Resume requires a training checkpoint and learning mode"
                )
            if self.physics_backend != "mujoco_warp":
                raise ValueError(
                    "Episode-boundary resume currently requires MuJoCo Warp"
                )
        if self.resume_compatibility_file is not None:
            if self.resume is None or not self.resume_compatibility_file.is_file():
                raise ValueError("Resume compatibility audit requires an existing checkpoint and audit file")
