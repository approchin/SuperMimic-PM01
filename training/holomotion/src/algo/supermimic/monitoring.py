# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Measured training counters shared by the reproduction and PPO control.

Episode summaries describe sampled training episodes, rather than the fixed
complete-clip evaluation population. Keep their names and denominators distinct.
"""

import math
import statistics
from collections import defaultdict


class TrainingRewardArea:
    """Integrate measured physical reward on a fixed training wall clock.

    Each observation is the mean reward of one real vector step. Hold that
    value until the next observation. Before the first observation, the curve
    is zero, so startup cannot earn unmeasured reward. Finalization is excluded
    by ending the curve at the last completed training boundary.
    """

    def __init__(self, horizon_seconds=1800.0, tail_seconds=300.0):
        if not (
            math.isfinite(horizon_seconds)
            and math.isfinite(tail_seconds)
            and 0 < tail_seconds <= horizon_seconds
        ):
            raise ValueError("Reward area requires finite positive durations")
        self.horizon_seconds = float(horizon_seconds)
        self.tail_seconds = float(tail_seconds)
        self.points = []

    def observe(self, training_seconds, mean_real_step_reward):
        time, reward = float(training_seconds), float(mean_real_step_reward)
        if not math.isfinite(time) or time < 0 or not math.isfinite(reward):
            raise ValueError(
                "Reward observations require finite time and reward"
            )
        if self.points and time <= self.points[-1][0]:
            raise ValueError("Reward observation times must strictly increase")
        self.points.append((time, reward))

    def summary(self, training_seconds):
        end = float(training_seconds)
        if not math.isfinite(end) or end < 0:
            raise ValueError("Reward summary requires a finite training clock")
        if self.points and end < self.points[-1][0]:
            raise ValueError("Reward summary cannot precede its observations")
        end = min(end, self.horizon_seconds)
        area = tail_area = 0.0
        tail_start = self.horizon_seconds - self.tail_seconds
        for index, (time, reward) in enumerate(self.points):
            right = min(
                self.points[index + 1][0]
                if index + 1 < len(self.points)
                else end,
                end,
            )
            area += reward * max(0.0, right - time)
            tail_area += reward * max(0.0, right - max(time, tail_start))
        complete = end == self.horizon_seconds and bool(self.points)
        return dict(
            metric="real_training_reward_wall_time_area_v1",
            source="Mean physical step reward across all training environments",
            integration="Previous observation held until the next; zero before the first",
            horizon_seconds=self.horizon_seconds,
            observed_training_seconds=end,
            complete=complete,
            real_vector_observations=len(self.points),
            first_observation_seconds=self.points[0][0]
            if self.points
            else None,
            reward_seconds=area,
            time_mean_reward=area / self.horizon_seconds if complete else None,
            final_window_seconds=self.tail_seconds,
            final_window_mean_reward=tail_area / self.tail_seconds
            if complete
            else None,
        )


class EpisodeMonitor:
    def __init__(self):
        self.transitions = self.terminated = self.ended = self.completed = 0
        self.length_sum = self.return_sum = 0.0

    def add(self, transition):
        self.transitions += transition["done"].numel()
        self.terminated += int(transition["term"].sum())
        self.ended += int(transition["done"].sum())
        if "ep_len" in transition:
            lengths, returns = transition["ep_len"], transition["ep_ret_norm"]
            if lengths.shape != returns.shape or len(lengths) != int(
                transition["done"].sum()
            ):
                raise ValueError(
                    "Episode monitoring requires one length and return per completed episode"
                )
            self.completed += len(lengths)
            self.length_sum += float(lengths.sum())
            self.return_sum += float(returns.sum())

    def summary(self, *, vector_steps, training_seconds, gpu_count=1):
        return dict(
            real_transitions=self.transitions,
            completed_episodes=self.completed,
            physical_terminations=self.terminated,
            physical_episode_endings=self.ended,
            termination_fraction_among_endings=self.terminated / self.ended
            if self.ended
            else None,
            sampled_episode_mean_length=self.length_sum / self.completed
            if self.completed
            else None,
            sampled_episode_mean_normalized_return=self.return_sum
            / self.completed
            if self.completed
            else None,
            real_transitions_per_training_second=self.transitions
            / training_seconds
            if training_seconds
            else None,
            vector_steps_per_training_second=vector_steps / training_seconds
            if training_seconds
            else None,
            training_gpu_hours=gpu_count * training_seconds / 3600,
            episode_population="Return-sampled training episodes; distinct from complete-clip evaluation",
        )


class DivergenceMonitor:
    """Repeated scale alarms request real inference; negative actor loss is valid."""

    def __init__(self, window=100, consecutive=3, factor=5.0):
        self.window, self.consecutive, self.factor = window, consecutive, factor
        self.values = defaultdict(list)
        self.baselines = defaultdict(list)
        self.bad = defaultdict(int)
        self.alerted = False

    def observe(self, events):
        alarms = []
        for event in events:
            module, metrics = event["module"], event["metrics"]
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                    raise FloatingPointError(f"Nonfinite logged scalar {module}/{name}")
            candidates = {}
            if module in ("critic", "critic_summary"):
                for name in ("loss", "td_mse", "critic_loss_mean"):
                    if name in metrics:
                        candidates[f"critic/{name}"] = metrics[name]
            if module in ("actor", "actor_summary"):
                for name in ("gradient_norm", "actor_gradient_norm", "grad_norm_preclip"):
                    if name in metrics:
                        candidates[f"actor/{name}"] = metrics[name]
                if "Jtail" in metrics and "Jreward" in metrics:
                    candidates["actor/tail_reward_ratio"] = abs(metrics["Jtail"]) / max(abs(metrics["Jreward"]), 0.01)
            for key, value in candidates.items():
                if not isinstance(value, (int, float)) or value < 0:
                    continue
                bucket = self.values[key]
                bucket.append(float(value))
                if len(bucket) < self.window:
                    continue
                measured = statistics.median(bucket)
                self.values[key] = []
                baseline_windows = self.baselines[key]
                if len(baseline_windows) < 3:
                    baseline_windows.append(max(measured, 1e-6))
                    continue
                baseline = max(statistics.median(baseline_windows), 1e-6)
                abnormal = measured > self.factor * baseline
                self.bad[key] = self.bad[key] + 1 if abnormal else 0
                if not abnormal:
                    baseline_windows.append(max(measured, 1e-6))
                    del baseline_windows[:-10]
                if self.bad[key] >= self.consecutive and not self.alerted:
                    alarms.append(dict(metric=key, window_median=measured, healthy_baseline=baseline,
                                       factor=self.factor, consecutive_windows=self.bad[key],
                                       action="request_fast_real_inference; actor_loss_sign_is_not_a_gate"))
        if alarms:
            self.alerted = True
        return alarms


class PushRecoveryCurriculum:
    """Approved PM01 two-stage reset supplement; never inject training impulses."""

    def __init__(self, state=None):
        self.stage, self.stage_since = 0, 0
        self.failed_push_rounds = self.failed_core_rounds = 0
        self.seen_jobs = set()
        if state:
            self.__dict__.update(state)
            self.seen_jobs = set(self.seen_jobs)

    @property
    def fraction(self):
        return (0.0, 0.1, 0.2)[self.stage]

    def state_dict(self):
        return dict(self.__dict__, seen_jobs=sorted(self.seen_jobs))

    def observe(self, result, *, current_step, diverging=False):
        if result["job_id"] in self.seen_jobs or result.get("reason") != "full":
            return None
        self.seen_jobs.add(result["job_id"])
        if int(result.get("vector_steps", current_step)) < self.stage_since:
            return None  # Late asynchronous results came from the preceding reset distribution.
        suites = result.get("suites", {})
        fixed, special, push = suites.get("fixed200"), suites.get("target18"), suites.get("push120")
        if not all((fixed, special, push)) or diverging:
            return None
        groups = special.get("groups", {})
        core_ok = (fixed.get("completed_clips", 0) >= 197
                   and groups.get("run_or_transition", {}).get("completed", 0) >= 4
                   and groups.get("jump", {}).get("completed", 0) >= 11)
        push_ok = push.get("light_completed", 0) >= 54 and push.get("medium_completed", 0) >= 48
        self.failed_core_rounds = 0 if core_ok else self.failed_core_rounds + 1
        self.failed_push_rounds = self.failed_push_rounds + 1 if core_ok and not push_ok else 0
        old = self.stage
        if self.failed_core_rounds >= 2 and self.stage > 0:
            self.stage -= 1
        elif (core_ok and self.failed_push_rounds >= 2 and self.stage < 2
              and (self.stage == 0 or current_step - self.stage_since >= 20000)):
            self.stage += 1
        if self.stage != old:
            self.stage_since = int(current_step)
            self.failed_core_rounds = self.failed_push_rounds = 0
            return dict(old_stage=old, new_stage=self.stage, stronger_reset_fraction=self.fraction,
                        vector_steps=current_step, model_rollback=False, seed_unchanged=True,
                        interpretation="replace xy offset with uniform direction, speed U[0.5,1.0] in selected resets; other resets unchanged")
        if self.stage == 2 and self.failed_push_rounds >= 2:
            return dict(stage=2, stronger_reset_fraction=0.2, action="report_still_below_push_target; do_not_escalate")
        return None
