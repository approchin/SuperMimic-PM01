# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Deterministic complete-clip scoring for the small reproduction benchmark."""

import time

import torch

from holomotion.src.env.reference_tracking.mdp import MAX_STEP_REWARD


def full_clip_scores(returns, planned_transitions):
    if (
        returns.shape != planned_transitions.shape
        or (planned_transitions <= 0).any()
        or not torch.isfinite(returns).all()
    ):
        raise ValueError(
            "Complete-clip scores require finite returns and positive planned lengths"
        )
    return returns / (MAX_STEP_REWARD * planned_transitions)


class FullClipEvaluator:
    def __init__(
        self, model, motion, mdp, *, device, history_length=10,
        future_frames=10, capture_graph=True, evaluation_mode="native",
        max_envs=64, dataset_role="training",
    ):
        from holomotion.src.env.reference_tracking.warp_backend import WarpTrackingBackend

        if type(max_envs) is not int or max_envs < 1:
            raise ValueError("Evaluation max_envs must be a positive integer")
        if not motion.names:
            raise ValueError("Evaluation requires at least one complete clip")
        if dataset_role not in ("training", "validation"):
            raise ValueError("Unknown evaluation dataset role")
        if evaluation_mode == "canonical_gpu_v1":
            if mdp.robot_spec.kind != "g1":
                raise ValueError("Canonical G1 evaluator is not a PM01 evaluation protocol")
            from .canonical_gpu.backend import CanonicalEvaluationBackend
            backend = CanonicalEvaluationBackend
        elif evaluation_mode == "native":
            backend = WarpTrackingBackend
        else:
            raise ValueError(f"Unknown evaluation mode: {evaluation_mode}")
        self.evaluation_mode = evaluation_mode
        self.dataset_role = dataset_role
        self.motion, self.mdp = motion, mdp
        self.history_length, self.future_frames = history_length, future_frames
        self.env = backend(
            model, motion, mdp, min(max_envs, len(motion.names)),
            device=device, evaluation=True, capture_graph=capture_graph,
        )

    @torch.no_grad()
    def evaluate(self, policy, *, progress=None):
        start = time.monotonic()
        motion, mdp, env = self.motion, self.mdp, self.env
        count = len(motion.names)
        planned = motion.clip_len - 1
        returns = torch.zeros(count, device=env.device)
        elapsed = torch.zeros(count, dtype=torch.long, device=env.device)
        failed = torch.zeros(count, dtype=torch.bool, device=env.device)
        body_error_sum = torch.zeros_like(returns)
        root_error_sum = torch.zeros_like(returns)
        trace = {
            name: [] for name in (
                "clip_id", "reference_frame", "previous_reference_frame",
                "previous_state", "next_state", "action", "reward",
                "terminated", "ended", "controller_time_s",
            )
        }
        world_ids = torch.arange(env.N, device=env.device)
        for first in range(0, count, env.N):
            selected = torch.arange(first, min(first + env.N, count), device=env.device)
            size = len(selected)
            assigned = torch.cat((selected, selected[-1:].expand(env.N - size)))
            env.done_eval.zero_()
            env.failed_eval.zero_()
            env.reset(world_ids, motion.clip_start[assigned])
            env.done_eval[size:] = True
            hist = mdp.obs(env.state(), env.a_prev, env.xi, env.f)[:, None].repeat(
                1, self.history_length, 1
            )
            batch_return = torch.zeros(env.N, device=env.device)
            batch_elapsed = torch.zeros(env.N, dtype=torch.long, device=env.device)
            batch_body_error = torch.zeros_like(batch_return)
            batch_root_error = torch.zeros_like(batch_return)
            max_steps = int(planned[selected].max())
            for step in range(max_steps):
                active = ~env.done_eval
                observation = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
                if step:
                    hist = torch.cat((hist[:, 1:], observation[:, None]), 1)
                action = policy(hist, motion.goal(env.f, self.future_frames))
                transition = env.step(action)
                packet = dict(
                    clip_id=assigned,
                    reference_frame=env.f,
                    previous_reference_frame=transition["f"],
                    previous_state=transition["s"],
                    next_state=transition["s1"],
                    action=transition["a"],
                    reward=transition["r"],
                    terminated=transition["term"],
                    ended=transition["done"],
                    controller_time_s=(transition["ep_step"] + 1).float() * 0.02,
                )
                for name, value in packet.items():
                    trace[name].append(value[active].cpu())
                batch_return += active * transition["r"]
                batch_elapsed += active
                state = env.state()
                batch_root_error += active * (state[0] - motion.root_p[env.f]).norm(dim=-1)
                bp = mdp.kin.forward(*state)[0]
                batch_body_error += active * (bp - motion.body_p[env.f]).norm(dim=-1).mean(-1)
                if progress is not None and (step + 1) % 1000 == 0:
                    progress(dict(
                        evaluation_step=step + 1,
                        completed_clips=first + int(env.done_eval[:size].sum()),
                        total_clips=count, parallel_envs=env.N,
                    ))
                if env.done_eval.all():
                    break
            if not env.done_eval[:size].all():
                raise RuntimeError("Complete-clip evaluator did not finish every planned trial")
            returns[selected] = batch_return[:size]
            elapsed[selected] = batch_elapsed[:size]
            failed[selected] = env.failed_eval[:size]
            body_error_sum[selected] = batch_body_error[:size]
            root_error_sum[selected] = batch_root_error[:size]
        scores = full_clip_scores(returns, planned)
        self.last_trace = {name: torch.cat(values) for name, values in trace.items()}
        self.last_trace.update(
            clip_names=motion.names,
            clip_start=motion.clip_start.cpu(),
            clip_length=motion.clip_len.cpu(),
            state_convention=f"root XYZ, WXYZ, world link linear/angular velocities, {mdp.robot_spec.joints} named joints and velocities",
            control_interval_s=0.02, dataset_role=self.dataset_role,
        )
        clips = [
            dict(
                name=name, normalized_return=float(scores[i]),
                total_reward=float(returns[i]), failed=bool(failed[i]),
                planned_steps=int(planned[i]), executed_steps=int(elapsed[i]),
                mean_global_tracked_body_error_mm=float(1000 * body_error_sum[i] / elapsed[i].clamp_min(1)),
                mean_root_position_error_mm=float(1000 * root_error_sum[i] / elapsed[i].clamp_min(1)),
            )
            for i, name in enumerate(motion.names)
        ]
        protocol = {"mode": "native"}
        if self.evaluation_mode == "canonical_gpu_v1":
            protocol = env.canonical.metadata()
        elif mdp.robot_spec.kind == "pm01":
            protocol.update(
                threshold_formula_id="any2any_paper_completion_v1",
                root_z_m=0.25, tracked_body_z_m=0.30,
                full_SO3_angle_squared_rad2=0.8, natural_completion_required=True,
                complete_protocol_equivalence_established=False,
                temporal_alignment="Native frame0 initialization; post-state scores frame1..N-1, N-1 transitions",
            )
        return dict(
            evaluation_protocol=protocol,
            evaluation_mode=self.evaluation_mode,
            new_evaluation_protocol=self.evaluation_mode != "native",
            baseline_rebuild_required=self.evaluation_mode != "native",
            scope=f"{self.dataset_role.capitalize()} complete-clip evaluation; deterministic starts at frame zero",
            dataset_role=self.dataset_role,
            reward_denominator="0.1 multiplied by full planned transitions, including unexecuted failure tails",
            normalized_return=float(scores.mean()), failed_clips=int(failed.sum()),
            frame_weighted_normalized_return=float(returns.sum() / (MAX_STEP_REWARD * planned.sum())),
            evaluated_clips=count, parallel_envs=env.N, clips=clips,
            elapsed_s=time.monotonic() - start,
            observation_noise=False, reset_noise=False, domain_randomization=False,
            foot_friction=1.0, physics_status=env.status_summary(),
        )
