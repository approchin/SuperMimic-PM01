# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Persistent native-Warp actors: complete clips and paired physical pushes."""

import json
import math
import os
import socket
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import fields, is_dataclass
from pathlib import Path

from .evaluation_queue import EvaluationQueue, write_json
from .retrain_runtime import intervention


FORK_ROUTE = "single_native_prefix_in_place_fork_v1"
JOINT_ANGLE_FORMAT = "pm01_joint_angle_error_v1"


def joint_angle_record(absolute_sums, squared_sums, executed_frames, robot_spec):
    """Finalize one real clip; sums are in degrees over active executed frames."""
    names = list(robot_spec.joint_names)
    controlled = [names.index(name) for name in robot_spec.policy_joint_names]
    exogenous = [i for i in range(len(names)) if i not in controlled]
    if len(absolute_sums) != len(names) or len(squared_sums) != len(names):
        raise ValueError("Joint-error sums differ from the named robot state layout")
    per_mae = [value / executed_frames for value in absolute_sums] if executed_frames else None
    per_mse = [value / executed_frames for value in squared_sums] if executed_frames else None

    def subset(indices):
        if not indices or not executed_frames:
            return dict(mae_deg=None, rmse_deg=None)
        return dict(mae_deg=sum(per_mae[i] for i in indices) / len(indices),
                    rmse_deg=math.sqrt(sum(per_mse[i] for i in indices) / len(indices)))

    return dict(format=JOINT_ANGLE_FORMAT, executed_frames=executed_frames,
                **subset(controlled), all_joints=subset(range(len(names))), exogenous=subset(exogenous),
                per_joint_mae_deg=dict(zip(names, per_mae)) if per_mae is not None else None,
                per_joint_rmse_deg=dict(zip(names, map(math.sqrt, per_mse))) if per_mse is not None else None)


def joint_angle_summary(records, robot_spec):
    """Equal weight per executed clip; frame weighting is explicitly auxiliary."""
    names = list(robot_spec.joint_names)
    primary = list(robot_spec.policy_joint_names)
    exogenous = [name for name in names if name not in primary]
    measured = [row["joint_angle_error"] for row in records if row["joint_angle_error"]["executed_frames"]]
    frames = sum(row["executed_frames"] for row in measured)
    planned = sum(row["planned_steps"] for row in records)
    per_mae = {name: sum(row["per_joint_mae_deg"][name] for row in measured) / len(measured)
               for name in names} if measured else None
    per_mse = {name: sum(row["per_joint_rmse_deg"][name] ** 2 for row in measured) / len(measured)
               for name in names} if measured else None

    def subset(selected, key=None):
        if not selected or not measured:
            return dict(mae_deg=None, rmse_deg=None, frame_weighted_mae_deg=None, frame_weighted_rmse_deg=None)
        def value(row, metric):
            return row[key][metric] if key else row[metric]
        return dict(mae_deg=sum(per_mae[name] for name in selected) / len(selected),
                    rmse_deg=math.sqrt(sum(per_mse[name] for name in selected) / len(selected)),
                    mean_clip_rmse_deg=sum(value(row, "rmse_deg") for row in measured) / len(measured),
                    frame_weighted_mae_deg=sum(value(row, "mae_deg") * row["executed_frames"] for row in measured) / frames,
                    frame_weighted_rmse_deg=math.sqrt(sum(value(row, "rmse_deg") ** 2 * row["executed_frames"] for row in measured) / frames))

    return dict(format=JOINT_ANGLE_FORMAT, unit="degree", joint_names=names,
                primary_joint_names=primary, primary_joint_count=len(primary), exogenous_joint_names=exogenous,
                state_alignment="unflatten_state(transition.s1)[4] versus motion.q[min(transition.f+1, clip_end)]",
                angular_difference="raw simulated q minus reference q; no angle wrapping for bounded hinges",
                frame_inclusion="active before step; includes terminating step; excludes padding and all post-done neutral steps; no unexecuted-frame zero fill",
                aggregation="mean over each clip's executed frames and primary joints, then equal weight per measured clip",
                rmse_formula="sqrt(mean_clip(mean_executed_frame(mean_primary_joint(error_deg**2))))",
                evaluated_clips=len(records), measured_clips=len(measured),
                completed_clips=sum(row["completed"] for row in records),
                completion_rate=sum(row["completed"] for row in records) / len(records) if records else None,
                executed_frames=frames, planned_frames=planned, executed_frame_fraction=frames / planned if planned else None,
                **subset(primary), all_joints=subset(names, "all_joints"), exogenous=subset(exogenous, "exogenous"),
                per_joint_mae_deg=per_mae,
                per_joint_rmse_deg={name: math.sqrt(value) for name, value in per_mse.items()} if per_mse else None)


def validate_condition_cases(cases, *, controls=False):
    required = {"condition_id", "clip_id", "frame", "delay_ms", "direction",
                "magnitude_mps", "expected_world_delta_v_xy_mps"}
    if not isinstance(cases, list) or not cases:
        raise ValueError("An explicit perturbation plan must contain nonempty case lists")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not required <= case.keys():
            raise ValueError(f"Perturbation case lacks required fields: {sorted(required - case.keys()) if isinstance(case,dict) else case}")
        if case["condition_id"] in ids:
            raise ValueError("Repeated perturbation condition_id")
        ids.add(case["condition_id"])
        if (type(case["frame"]) is not int or case["frame"] < 0
                or type(case["delay_ms"]) is not int or case["delay_ms"] not in (0,5,10,15,20)
                or case["direction"] not in ("forward", "backward", "left", "right")):
            raise ValueError(f"Invalid perturbation frame/delay/direction: {case['condition_id']}")
        magnitude, vector = case["magnitude_mps"], case["expected_world_delta_v_xy_mps"]
        if (isinstance(magnitude, bool) or not isinstance(magnitude, (int,float)) or not math.isfinite(magnitude)
                or magnitude < 0 or (controls and magnitude != 0)
                or not isinstance(vector, (list,tuple)) or len(vector) != 2
                or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in vector)
                or not math.isclose(math.hypot(*vector), magnitude, rel_tol=1e-5, abs_tol=1e-6)):
            raise ValueError(f"Invalid perturbation magnitude/world vector: {case['condition_id']}")


def paired_controls(cases, controls):
    validate_condition_cases(cases)
    validate_condition_cases(controls, controls=True)
    if any(case["magnitude_mps"] <= 0 for case in cases):
        raise ValueError("Push-plan primary cases must prescribe a nonzero impulse")
    by_control = {case["condition_id"]:case for case in controls}
    paired_ids = set()
    for case in cases:
        other = by_control.get(case.get("no_push_control_id"))
        if (other is None or other.get("paired_push_condition_id") != case["condition_id"]
                or any(other[key] != case[key] for key in ("clip_id", "frame", "delay_ms", "direction"))):
            raise ValueError(f"Push/control IDs or clip/frame/delay/direction are not paired: {case['condition_id']}")
        paired_ids.add(other["condition_id"])
    if len(paired_ids) != len(controls) or len(cases) != len(controls):
        raise ValueError("Push controls are not a one-to-one pairing")
    return by_control


def load_condition_plan(suite):
    if not suite.get("conditions"):
        return None, None
    plan = json.loads(Path(suite["conditions"]).read_text())
    cases = plan[suite.get("conditions_key", "conditions")]
    controls = plan[suite.get("controls_key", "no_push_controls")]
    paired_controls(cases, controls)
    if (len(cases) != suite.get("planned_conditions", len(cases))
            or len(controls) != suite.get("controls", len(cases))):
        raise ValueError("Perturbation plan/control denominators disagree with the registered suite")
    if suite.get("suite_id") == "push120":
        if (len(cases) != 120 or Counter(c["magnitude_mps"] for c in cases) != {.5:60,1.:60}
                or Counter((c["magnitude_mps"],c["delay_ms"]) for c in cases)
                   != {(m,d):12 for m in (.5,1.) for d in (0,5,10,15,20)}):
            raise ValueError("push120 requires 120 explicit cases, 60 per strength and 12 per strength/delay")
    return cases, controls


def uses_delay_sweep(suite, conditions):
    # A delay list on a push suite describes case stratification. It must
    # never replace the explicit timed impulses with a no-push sweep.
    return conditions is None and isinstance(suite.get("delay_ms"), list)


def qualified_push120(result):
    return bool(result and result.get("perturbation_route") == FORK_ROUTE
                and result.get("valid_for_push_metrics") is True
                and result.get("shared_prefix_verified_pairs") == 120
                and result.get("light_denominator") == result.get("medium_denominator") == 60)


def annotate_legacy_push_results(queue, run_root):
    """Keep original JSON, annotate wrong-route evidence and invalidate derived bests."""
    import fcntl
    invalid = {}
    for path in (queue.root / "results").glob("*.json"):
        packet = json.loads(path.read_text())
        push = packet.get("suites", {}).get("push120")
        if not push or qualified_push120(push):
            continue
        reason = ("legacy_delay_sweep_replaced_timed_push_conditions" if "by_delay_ms" in push
                  else "legacy_missing_verified_shared_physical_prefix")
        invalid[packet["job_id"]] = dict(valid_for_push_metrics=False,
            reason=reason + "; no_antipush_success_or_failure_claim",
            original_result=str(path), original_subsuite=str(Path(run_root) / "evaluations" / packet["job_id"] / "push120.json"),
            recorded_evaluated_clips=push.get("evaluated_clips"), required_conditions=120,
            original_json_preserved=True)
        write_json(Path(run_root) / "evaluation_validity" / f"{packet['job_id']}_push120.json", invalid[packet["job_id"]])
    if not invalid:
        return
    with (queue.root / "best.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = queue.root / "best.json"
        if path.exists():
            best = json.loads(path.read_text())
            for label in list(best):
                if label.startswith("push_") and best[label].get("job_id") in invalid:
                    entry = best.pop(label)
                    best.setdefault("invalidated_push_bests", {})[label] = dict(entry, valid_for_push_metrics=False,
                        invalid_reason=invalid[entry["job_id"]]["reason"])
                    queue.protect(entry["actor"], "invalid_push_route_evidence")
            write_json(path, best)
        fcntl.flock(lock, fcntl.LOCK_UN)


class NativeFork:
    """Snapshot the entire batch, including cross-world contact and solver storage."""

    def __init__(self, batch):
        import torch
        import warp as wp

        self.arrays, self.scalars, self.tensors = [], [], []

        def visit(node, path):
            # Include instance attributes in addition to the pinned native schema.
            names = dict.fromkeys([f.name for f in fields(node)] + list(vars(node)))
            for name in names:
                value = getattr(node, name)
                key = f"{path}.{name}"
                if isinstance(value, wp.array):
                    tensor = wp.to_torch(value) if value.size else None
                    self.arrays.append((node, name, key, value, value.ptr, tensor))
                    if tensor is not None:
                        self.tensors.append(("data", key, tensor, tensor.clone()))
                elif is_dataclass(value):
                    visit(value, key)
                elif value is None or isinstance(value, (int, float, bool, str)):
                    self.scalars.append((node, name, value))
                else:
                    raise TypeError(f"Unsupported mutable native field {key}: {type(value)}")

        visit(batch.env.data, "data")
        self.bindings = []
        for group, owner in (("backend", batch.env), ("rollout", batch)):
            for name, value in vars(owner).items():
                if torch.is_tensor(value):
                    self.bindings.append((owner, name, value))
                    self.tensors.append((group, name, value, value.clone()))

    def restore_and_verify(self):
        import torch

        # CUDA graph capture retains these addresses. Never swap a native Data
        # or its arrays. The backend rebinds f, ep_step and a_prev at each step.
        for node, name, key, array, ptr, _ in self.arrays:
            if getattr(node, name) is not array or array.ptr != ptr:
                raise RuntimeError(f"Native array binding changed during rollout: {key}")
        for owner, name, value in self.bindings:
            if getattr(owner, name) is not value:
                setattr(owner, name, value)
        for owner, name, value in self.scalars:
            setattr(owner, name, value)
        for _, _, live, saved in self.tensors:
            live.copy_(saved)

        # Inspect storage bytes so a NaN or signed zero in unused solver workspace
        # cannot mask a failed restore. Use one synchronization for all comparisons.
        mismatches = []
        for _, _, live, saved in self.tensors:
            left = live.contiguous().reshape(-1).view(torch.uint8)
            right = saved.contiguous().reshape(-1).view(torch.uint8)
            mismatches.append(torch.count_nonzero(left != right))
        counts = torch.stack(mismatches).cpu().tolist()
        different = {entry[1]: n for entry, n in zip(self.tensors, counts) if n}
        if different or any(getattr(owner, name) != value for owner, name, value in self.scalars):
            raise RuntimeError(f"Shared prefix restore failed the native/rollout readback: {different}")
        return dict(native_data_arrays=len(self.arrays), native_data_scalar_fields=len(self.scalars),
                    native_data_bytes_compared=sum(live.numel() * live.element_size()
                        for group, _, live, _ in self.tensors if group == "data"),
                    backend_tensors_compared=sum(t[0] == "backend" for t in self.tensors),
                    rollout_tensors_compared=sum(t[0] == "rollout" for t in self.tensors),
                    byte_mismatches=sum(counts), native_array_bindings_preserved=True)


def compare_control_prefixes(result, control_result, forks):
    import numpy as np

    by_control = {row["condition_id"]: row for row in control_result["clips"]}
    comparisons = []
    for row in result["clips"]:
        control = by_control[row["no_push_control_id"]]
        reached = row["event_reached"]
        if reached != control["event_reached"]:
            raise RuntimeError(f"Push/control event availability differs: {row['condition_id']}")
        differences = None
        if reached:
            a, b = row["pre_event_state"], control["pre_event_state"]
            if a.keys() != b.keys():
                raise RuntimeError("Push/control event state fields differ")
            differences = {}
            for key in a:
                left, right = np.asarray(a[key], dtype=np.float64), np.asarray(b[key], dtype=np.float64)
                if left.shape != right.shape or not np.array_equal(left, right):
                    raise RuntimeError(f"Pre-event state differs for {row['condition_id']}: {key}")
                differences[key] = float(np.abs(left - right).max())
        elif (not row["pre_injection_failure"] or not control["terminated"]
              or any(row[key] != control[key] for key in ("executed_steps", "total_reward", "mean_root_error_m", "mean_body_error_m"))):
            raise RuntimeError(f"Expected the identical shared pre-event termination: {row['condition_id']}")
        comparisons.append(dict(condition_id=row["condition_id"], control_id=control["condition_id"],
            actual_pre_event_comparison_available=reached, pre_event_max_abs_difference=differences,
            actual_pre_event_values_equal=reached, shared_termination_before_event=not reached))
    fork_ids = [key for fork in forks for key in fork["condition_ids"]]
    if Counter(fork_ids) != Counter(row["condition_id"] for row in result["clips"]):
        raise RuntimeError("Shared prefix fork evidence does not cover the evaluated cases")
    result.update(control_pair_evidence=comparisons, shared_prefix_forks=forks,
                  shared_prefix_verified_pairs=len(comparisons),
                  control_pairing_scope="one native prefix per group; same batch and lane restored in place at the event")


def _group(item):
    item = item.get("metadata", item)
    label = item.get("sampling_group", item.get("primary_label", item.get("label", "ordinary")))
    tier = item.get("reference_tier", "")
    if "challenge" in tier or "challenge" in label:
        return "challenge_run"
    if label in ("run", "run_or_transition", "conservative_run_or_transition"):
        return "run_or_transition"
    if "jump" in label:
        return "jump"
    return "ordinary"


class TrialBatch:
    """One reset and its native policy/physics timeline, including metric accumulators."""

    def __init__(self, evaluator, cases):
        torch, env, motion = evaluator.torch, evaluator.env, evaluator.motion
        self.evaluator, self.env = evaluator, env
        padded = cases + [cases[-1]] * (env.N - len(cases))
        self.assigned = torch.tensor([evaluator.names[c["clip_id"]] for c in padded], device=env.device)
        self.starts = motion.clip_start[self.assigned]
        self.planned = motion.clip_len[self.assigned] - 1
        self.horizon = int(self.planned[:len(cases)].max())
        ids = torch.arange(env.N, device=env.device)
        env.done_eval.zero_()
        env.failed_eval.zero_()
        env.reset(ids, self.starts)
        evaluator._set_delays(ids, torch.tensor([c.get("delay_ms", 0) for c in padded], device=env.device))
        env.done_eval[len(cases):] = True
        observation = evaluator.mdp.obs(env.state(), env.a_prev, env.xi, env.f)
        self.history = observation[:, None].repeat(1, evaluator.hist, 1)
        self.returns = torch.zeros_like(env.ep_ret)
        self.elapsed = torch.zeros_like(env.f)
        self.root_error = torch.zeros_like(self.returns)
        self.body_error = torch.zeros_like(self.returns)
        # Named native q layout; keep all sums on device until this batch ends.
        self.joint_absolute_error = torch.zeros((env.N, evaluator.mdp.robot_spec.joints),
                                                device=env.device, dtype=torch.float64)
        self.joint_squared_error = torch.zeros_like(self.joint_absolute_error)
        self.reached = torch.zeros_like(env.done_eval)
        self.applied_dv = torch.zeros_like(env.qvel[:, :3])
        self.event_frames = torch.full_like(env.f, -1)
        self.stable = torch.zeros_like(env.f)
        self.recovered_at = torch.full_like(self.returns, float("inf"))
        self.confirmed_at = torch.full_like(self.recovered_at, float("inf"))

    def event(self, cases, frame):
        import warp as wp

        torch, env = self.evaluator.torch, self.env
        self.reached.copy_(~env.done_eval)
        local_frames = env.f - self.starts
        if bool((self.reached & (local_frames != frame)).any()):
            raise RuntimeError("An active world reached the fork on a different reference frame")
        self.event_frames[self.reached] = local_frames[self.reached]
        state = dict(qpos=env.qpos, qvel=env.qvel, qacc_warmstart=wp.to_torch(env.data.qacc_warmstart),
                     time=wp.to_torch(env.data.time), history=self.history, delay_state=env.delay_state,
                     a_prev=env.a_prev, issued_target=env._issued_target, ctrl=env.ctrl, xi=env.xi,
                     reference_frame=env.f, episode_step=env.ep_step, done=env.done_eval, failed=env.failed_eval)
        # Read each branch from the live state immediately before its own impulse.
        pre_event = {name: value[:len(cases)].cpu().tolist() for name, value in state.items()}
        padded = cases + [cases[-1]] * (env.N - len(cases))
        xy = env.qvel.new_tensor([c["expected_world_delta_v_xy_mps"] for c in padded])
        before = env.qvel[:, :3].clone()
        if any(c["magnitude_mps"] > 0 for c in cases):
            env.qvel[self.reached, :2] += xy[self.reached]
        self.applied_dv.copy_(env.qvel[:, :3] - before)
        return pre_event

    def advance(self, begin, end, trace):
        from holomotion.src.env.reference_tracking.mdp import unflatten_state

        ev, env = self.evaluator, self.env
        torch, motion, mdp = ev.torch, ev.motion, ev.mdp
        if begin >= end or bool(env.done_eval.all()):
            return
        for step in range(begin, end):
            active = ~env.done_eval
            observation = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
            self.history = (torch.cat((self.history[:, 1:], observation[:, None]), 1) if step
                            else observation[:, None].repeat(1, ev.hist, 1))
            action = ev.policy(self.history, motion.goal(env.f, ev.future))
            transition = env.step(action)
            self.returns += active * transition["r"]
            self.elapsed += active
            next_frame = torch.minimum(transition["f"] + 1, motion.clip_end[motion.clip_id[transition["f"]]])
            error_deg = (unflatten_state(transition["s1"])[4] - motion.q[next_frame]).to(torch.float64) * (180. / math.pi)
            error_deg = torch.where(active[:, None], error_deg, 0.)
            self.joint_absolute_error += error_deg.abs()
            self.joint_squared_error += error_deg.square()
            state = env.state()
            self.root_error += active * (state[0] - motion.root_p[env.f]).norm(dim=-1)
            self.body_error += active * (mdp.kin.forward(*state)[0] - motion.body_p[env.f]).norm(dim=-1).mean(-1)
            if ev.conditioned:
                settled = (state[2][:, :2].norm(dim=-1) < .1) & (state[3].norm(dim=-1) < .2)
                self.stable = torch.where(self.reached & active & settled, self.stable + 1, torch.zeros_like(self.stable))
                now = (env.f - self.starts - self.event_frames).float() * .02
                found = self.reached & (self.stable >= 25) & ~torch.isfinite(self.recovered_at)
                self.recovered_at = torch.where(found, (now - .5).clamp_min(0.), self.recovered_at)
                self.confirmed_at = torch.where(found, now, self.confirmed_at)
            if trace is not None:
                trace.append(torch.cat((self.assigned[:, None].float(), transition["f"][:, None].float(),
                                        active[:, None].float(), transition["s1"]), -1).detach())
            if bool(env.done_eval.all()):
                break

    def records(self, cases, pre_event=None):
        torch = self.evaluator.torch
        scores = self.returns / (.1 * self.planned)
        summary = torch.stack((scores, self.returns, self.elapsed, self.env.failed_eval, self.reached,
                               self.recovered_at, self.root_error / self.elapsed.clamp_min(1),
                               self.body_error / self.elapsed.clamp_min(1), self.confirmed_at,
                               self.event_frames, *self.applied_dv.unbind(1), self.planned), 1).cpu().tolist()
        joint_sums = torch.stack((self.joint_absolute_error, self.joint_squared_error), 1).cpu().tolist()
        records = []
        for i, case in enumerate(cases):
            score, reward, length, failed, reached, recovery, root, body, confirmation, frame, vx, vy, vz, planned = summary[i]
            magnitude = case.get("magnitude_mps", 0.)
            if reached:
                expected = [*case["expected_world_delta_v_xy_mps"], 0.]
                if (int(frame) != case["frame"] or
                        (magnitude == 0 and any(v != 0 for v in (vx, vy, vz))) or
                        any(not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                            for a, b in zip((vx, vy, vz), expected))):
                    raise RuntimeError(f"Physical impulse readback disagrees with plan: {case['condition_id']}")
            records.append(dict(case, normalized_return=score, total_reward=reward,
                planned_steps=int(planned), executed_steps=int(length),
                completed=not bool(failed) and length == planned and (not self.evaluator.conditioned or bool(reached)),
                terminated=bool(failed), event_reached=bool(reached), actual_event_frame=int(frame) if reached else None,
                actual_injection=bool(reached) and magnitude > 0, actual_world_delta_v_xy_mps=[vx, vy],
                pre_event_state={name: values[i] for name, values in pre_event.items()} if reached else None,
                pre_injection_failure=bool(failed) and magnitude > 0 and not bool(reached),
                recovery_time_s=recovery if math.isfinite(recovery) else None,
                recovery_confirmation_time_s=confirmation if math.isfinite(confirmation) else None,
                recovered_within_5s=math.isfinite(confirmation) and confirmation <= 5.00001,
                mean_root_error_m=root, mean_body_error_m=body, group=self.evaluator.labels.get(case["clip_id"], "ordinary"),
                joint_angle_error=joint_angle_record(*joint_sums[i], int(length), self.evaluator.mdp.robot_spec)))
        return records


class CachedEvaluator:
    def __init__(self, suite, arguments, cache_dir, device, max_envs=64):
        import torch
        from holomotion.src.env.reference_tracking.task import load_task
        from holomotion.src.env.reference_tracking.warp_backend import WarpTrackingBackend
        from .networks import Policy
        self.torch, self.device, self.suite = torch, device, suite
        self.hist, self.future = arguments["hist"], arguments["goal_frames"]
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model, self.motion, self.mdp = load_task(
            robot_urdf=Path(arguments["robot_urdf"]), manifest=Path(suite["manifest"]),
            output=cache_dir, device=device, profile="paper", robot_spec=Path(arguments["robot_spec"]))
        self.names = {name: i for i, name in enumerate(self.motion.names)}
        self.conditioned = bool(suite.get("conditions"))
        case_plan, _ = load_condition_plan(suite)
        planned_lanes = (max(Counter(case["frame"] for case in case_plan).values())
                         if case_plan is not None else len(self.motion.names))
        if case_plan is not None:
            lengths = self.motion.clip_len.cpu().tolist()
            for case in case_plan:
                if case["clip_id"] not in self.names or case["frame"] >= lengths[self.names[case["clip_id"]]] - 1:
                    raise ValueError(f"Unknown clip or push frame outside physical transitions: {case['condition_id']}")
        self.env = WarpTrackingBackend(self.model, self.motion, self.mdp,
                                      min(max_envs, max(planned_lanes, 1)),
                                      device=device, evaluation=True, capture_graph=arguments.get("capture_graph", True),
                                      eval_delay_ms=int(suite.get("delay_ms", 0)) if isinstance(suite.get("delay_ms", 0), int) else 0)
        self.policy = Policy(self.mdp.obs_dim, self.mdp.robot_spec.goal_frame_dim * self.future,
                             hist=self.hist, action_dim=self.mdp.robot_spec.actions).to(device)
        self.policy.eval()
        metadata = json.loads(Path(suite["manifest"]).read_text())
        self.labels = {}
        clips = metadata.get("clips", metadata.get("motions", []))
        if isinstance(clips, dict):
            clips = [dict(value, clip_id=key) for key, value in clips.items()]
        for item in clips:
            if isinstance(item, dict):
                self.labels[item.get("clip_id", item.get("name"))] = _group(item)
        # The 18-clip manifest may retain labels in a selection sidecar.
        for key in ("selection", "selection_file", "metadata_file"):
            if suite.get(key):
                records = json.loads(Path(suite[key]).read_text()).get("clips", [])
                self.labels.update({item["clip_id"]: _group(item) for item in records})

    def _set_delays(self, ids, values):
        if hasattr(self.env, "set_eval_delay"):
            self.env.set_eval_delay(ids, values)
        elif getattr(self.env, "delay_state", None) is not None:
            self.env.delay_state[ids, -1] = values.to(self.env.delay_state.dtype) / 5
        elif bool((values != 0).any()):
            raise RuntimeError("Evaluation requested delay but native backend has no delay state")

    def evaluate(self, pi, *, conditions=None, save_trace=None):
        """Unperturbed fixed clips or a single fixed latency stratum."""
        if self.conditioned:
            raise ValueError("A conditioned suite must use evaluate_paired with its registered controls")
        if conditions is None:
            conditions = [dict(condition_id=clip, clip_id=clip, delay_ms=self.suite.get("delay_ms", 0), magnitude_mps=0)
                          for clip in self.motion.names]
        self.policy.load_state_dict(pi, strict=True)
        started = time.monotonic()
        records, traces = [], []
        with self.torch.no_grad():
            for first in range(0, len(conditions), self.env.N):
                cases = conditions[first:first + self.env.N]
                batch = TrialBatch(self, cases)
                trace = [] if save_trace is not None else None
                batch.advance(0, batch.horizon, trace)
                records.extend(batch.records(cases))
                if trace:
                    traces.append(self.torch.stack(trace).cpu())
        self._save_trace(save_trace, conditions, traces)
        return self._result(conditions, records, time.monotonic() - started)

    def evaluate_paired(self, pi, conditions, controls, *, save_trace=None, control_trace=None):
        """Run exactly one physical prefix for every pair on its original world lane."""
        if not self.conditioned:
            raise ValueError("Paired evaluation requires a registered condition plan")
        by_control = paired_controls(conditions, controls)
        self.policy.load_state_dict(pi, strict=True)
        groups = defaultdict(list)
        for case in conditions:
            groups[case["frame"]].append(case)
        records = [[], []]
        traces = [[], []]
        batch_ids = [[], []]
        elapsed = [0., 0.]
        forks = []
        started = time.monotonic()
        with self.torch.no_grad():
            for frame, group in groups.items():
                for first in range(0, len(group), self.env.N):
                    cases = group[first:first + self.env.N]
                    matched = [by_control[c["no_push_control_id"]] for c in cases]
                    prefix_started = time.monotonic()
                    batch = TrialBatch(self, cases)
                    prefix_trace = [] if save_trace is not None or control_trace is not None else None
                    batch.advance(0, frame, prefix_trace)
                    snapshot = NativeFork(batch)
                    prefix_seconds = time.monotonic() - prefix_started
                    for branch, (plan, path) in enumerate(((cases, save_trace), (matched, control_trace))):
                        branch_started = time.monotonic()
                        if branch == 1:
                            verification = snapshot.restore_and_verify()
                            forks.append(dict(verification, frame=frame,
                                condition_ids=[c["condition_id"] for c in cases]))
                        # The history still ends at the previous observation. The
                        # first branch-specific observation sees the applied impulse.
                        pre_event = batch.event(plan, frame)
                        trace = list(prefix_trace) if path is not None else None
                        batch.advance(frame, batch.horizon, trace)
                        records[branch].extend(batch.records(plan, pre_event))
                        if trace:
                            traces[branch].append(self.torch.stack(trace).cpu())
                            batch_ids[branch].append([c["condition_id"] for c in plan])
                        elapsed[branch] += prefix_seconds + time.monotonic() - branch_started
                    del snapshot
        def in_plan_order(rows, plan):
            by_id = {row["condition_id"]: row for row in rows}
            return [by_id[c["condition_id"]] for c in plan]

        self._save_trace(save_trace, conditions, traces[0], batch_ids[0])
        self._save_trace(control_trace, controls, traces[1], batch_ids[1])
        result = self._result(conditions, in_plan_order(records[0], conditions), elapsed[0])
        control_result = self._result(controls, in_plan_order(records[1], controls), elapsed[1])
        compare_control_prefixes(result, control_result, forks)
        result["no_push_controls"] = control_result
        result["paired_elapsed_s"] = time.monotonic() - started
        # A selected small GPU probe uses the same evaluator, but is not a full suite.
        registered, registered_controls = load_condition_plan(self.suite)
        result["valid_for_push_metrics"] = conditions == registered and controls == registered_controls
        return result

    @staticmethod
    def _save_trace(path, conditions, traces, batch_ids=None):
        if path is None:
            return
        from .checkpoint import atomic_torch_save
        packet = dict(format="pm01_011_physical_eval_trace_v1", batches=traces, conditions=conditions,
                      state_convention="clip index, global reference frame, active flag, native flattened next state",
                      allows_recovery_steps_and_worldXY_displacement=True)
        if batch_ids is not None:
            packet.update(format="pm01_011_paired_physical_eval_trace_v2", batch_condition_ids=batch_ids,
                          padding="unused trailing lanes in each batch are inactive")
        atomic_torch_save(packet, path)

    def _result(self, conditions, records, elapsed):
        result = dict(evaluated_clips=len(records), completed_clips=sum(r["completed"] for r in records),
                      normalized_return=sum(r["normalized_return"] for r in records) / max(len(records), 1),
                      clips=records, elapsed_s=elapsed,
                      physics_mode="native_mujoco_warp", actor_input="reference_only_v0", precision="native_bf16",
                      observation_noise=False, domain_randomization=False, dataset_role="fixed_regression_or_development",
                      termination="original PM01 rootZ0.25m/bodyZ0.30m/fullSO3sq0.8; no new position/joint gate")
        groups = {}
        for record in records:
            group = groups.setdefault(record["group"], dict(total=0, completed=0))
            group["total"] += 1; group["completed"] += int(record["completed"])
        result["groups"] = groups
        result["joint_angle_error"] = joint_angle_summary(records, self.mdp.robot_spec)
        if self.conditioned:
            result.update(planned_conditions=len(conditions),
                          condition_events_reached=sum(r["event_reached"] for r in records),
                          actual_injections=sum(r["actual_injection"] for r in records),
                          pre_injection_failures=sum(r["pre_injection_failure"] for r in records),
                          perturbation_route=FORK_ROUTE,
                          valid_for_push_metrics=False,
                          by_magnitude={str(m):dict(planned=sum(c["magnitude_mps"]==m for c in conditions),
                              completed=sum(r["completed"] for r in records if r["magnitude_mps"]==m),
                              actual_injections=sum(r["actual_injection"] for r in records if r["magnitude_mps"]==m),
                              pre_injection_failures=sum(r["pre_injection_failure"] for r in records if r["magnitude_mps"]==m))
                              for m in sorted({c["magnitude_mps"] for c in conditions})})
        if self.conditioned and any(c["magnitude_mps"] > 0 for c in conditions):
            result["light_completed"] = sum(r["completed"] for r in records if r.get("magnitude_mps") == .5)
            result["medium_completed"] = sum(r["completed"] for r in records if r.get("magnitude_mps") == 1.)
            result["light_denominator"] = sum(c["magnitude_mps"] == .5 for c in conditions)
            result["medium_denominator"] = sum(c["magnitude_mps"] == 1. for c in conditions)
        return result


def quick_decision(suites, baseline):
    current = suites.get("fixed25", {})
    if not baseline or not baseline.get("fixed25"):
        return dict(stop=True, reason="no_comparable_real_baseline; retain_snapshot_and_request_review")
    for source, label in ((suites, "current"), (baseline, "baseline")):
        fixed = source.get("fixed25", {})
        groups = source.get("target18", {}).get("groups", {})
        if (not all(key in fixed for key in ("completed_clips", "normalized_return"))
                or fixed.get("evaluated_clips", 25) != 25
                or any("completed" not in groups.get(group, {}) for group in ("run_or_transition", "jump"))):
            return dict(stop=True, reason=f"incomplete_{label}_real_assessment; retain_pause_for_review")
    old = baseline["fixed25"]
    completion_drop = int(old["completed_clips"]) - int(current["completed_clips"])
    return_drop = float(old["normalized_return"]) - float(current["normalized_return"])
    special_drops = {}
    for group in ("run_or_transition", "jump"):
        previous = baseline.get("target18", {}).get("groups", {}).get(group, {}).get("completed")
        latest = suites.get("target18", {}).get("groups", {}).get(group, {}).get("completed")
        if previous is not None and latest is not None:
            special_drops[group] = int(previous) - int(latest)
    bad = completion_drop >= 2 or return_drop >= .05 or any(value >= 1 for value in special_drops.values())
    return dict(stop=bad, reason="real_inference_regressed" if bad else "real_inference_not_regressed",
                fixed25_completion_drop=completion_drop, fixed25_return_drop=return_drop, special_drops=special_drops)


def update_bests(queue, job, suites):
    """Persist the useful distinct category winners and a single joint winner."""
    # Each worker has independent result writes; serialize the tiny best index.
    import fcntl
    with (queue.root / "best.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = queue.root / "best.json"
        best = json.loads(path.read_text()) if path.exists() else {}
        promoted = []
        fixed, special, push = suites.get("fixed200"), suites.get("target18"), suites.get("push120")
        scores = {}
        if fixed:
            scores["ordinary_best"] = [fixed["completed_clips"], fixed["normalized_return"]]
        if special:
            for group in ("run_or_transition", "jump"):
                value = special.get("groups", {}).get(group)
                if value:
                    scores[f"{group}_best"] = [value["completed"], special["normalized_return"]]
        if qualified_push120(push):
            light, medium = push["light_completed"], push["medium_completed"]
            scores["push_light_best"] = [light, push["normalized_return"]]
            scores["push_medium_best"] = [medium, push["normalized_return"]]
            old = best.get("push_joint_best", {}).get("score", [0, 0, 0])
            if light >= old[0] and medium >= old[1]:
                scores["push_joint_best"] = [light, medium, push["normalized_return"]]
        if fixed and special and fixed["completed_clips"] >= 197:
            groups = special.get("groups", {})
            run = groups.get("run_or_transition", {}).get("completed", 0)
            jump = groups.get("jump", {}).get("completed", 0)
            old = best.get("joint_best", {}).get("score", [0, 0, 0, 0])
            if run >= old[0] and jump >= old[1]:
                scores["joint_best"] = [run, jump, fixed["completed_clips"], fixed["normalized_return"]]
        for label, score in scores.items():
            if label not in best or tuple(score) > tuple(best[label]["score"]):
                best[label] = dict(actor=job["actor"], score=score, vector_steps=job["vector_steps"], job_id=job["job_id"])
                queue.assign(job["actor"], label)
                promoted.append(label)
        write_json(path, best)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return promoted


def effective_job_suites(queue, job):
    """Complete full rounds; explicit selected-best/final supplements run once."""
    requested = list(job.get("requested_suites", job["suites"]))
    additions = []
    if job["reason"] == "full":
        additions = ["static16", "fixed200_delay"]
    elif job["reason"] in ("best_candidate", "final"):
        covered, holdout_scheduled = set(), False
        with queue.locked():
            for state in ("pending", "running", "completed"):
                for path in (queue.root / state).glob("*.json"):
                    other = json.loads(path.read_text())
                    if other["job_id"] == job["job_id"]:
                        continue
                    suites = other.get("suites", [])
                    if other["actor"] == job["actor"]:
                        covered.update(suites)
                    holdout_scheduled |= "holdout40" in suites
        additions = [name for name in ("static16", "fixed200_delay", "target100", "holdout40")
                     if name not in covered and not (name == "holdout40" and holdout_scheduled)]
    return requested, list(dict.fromkeys([*requested, *additions]))


def worker_main(*, run_root, suites_path, device="cuda:0", max_envs=64, poll_seconds=.5, once=False):
    import torch
    from . import networks
    run_root = Path(run_root)
    queue = EvaluationQueue(run_root / "eval_queue")
    registry = json.loads(Path(suites_path).read_text())
    registry = registry.get("suites", registry)
    annotate_legacy_push_results(queue, run_root)
    torch.set_num_threads(4); torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    networks.AMP_DTYPE = torch.bfloat16
    torch.manual_seed(13)
    worker_id = f"{socket.gethostname()}_{os.getpid()}"
    cache = {}
    intervention(run_root, "evaluation_worker_started", worker=worker_id, device=device)
    while True:
        job = queue.claim(worker_id)
        if job is None:
            if once:
                return
            time.sleep(poll_seconds)
            continue
        try:
            with queue.locked():
                requested, effective = effective_job_suites(queue, job)
                job.update(requested_suites=requested, suites=effective,
                           added_suites=[name for name in effective if name not in requested])
                write_json(queue.root / "running" / f"{job['job_id']}.json", job)
            bundle = torch.load(job["actor"], map_location="cpu", weights_only=False)
            arguments = bundle["specification"]["arguments"]
            results = {}
            for name in job["suites"]:
                suite = registry[name]
                if name not in cache:
                    cache[name] = CachedEvaluator(suite, arguments, run_root / "evaluation_workers" / worker_id / name,
                                                  device, max_envs=max_envs)
                evaluator = cache[name]
                trace = None
                if job["reason"] in ("full", "divergence", "nonfinite_last_finite", "final", "best_trace", "push_eval_repair"):
                    trace = run_root / "evaluations" / job["job_id"] / f"{name}_trace.pt"
                conditions, controls = load_condition_plan(suite)
                if uses_delay_sweep(suite, conditions):
                    by_delay = {}
                    for delay in suite["delay_ms"]:
                        delay_cases = [dict(condition_id=f"{clip}__delay{delay}", clip_id=clip, delay_ms=delay, magnitude_mps=0)
                                       for clip in evaluator.motion.names]
                        delay_trace = trace.with_name(f"{name}_{delay}ms_trace.pt") if trace else None
                        by_delay[str(delay)] = evaluator.evaluate(bundle["pi"], conditions=delay_cases, save_trace=delay_trace)
                    result = dict(by_delay_ms=by_delay, comparison="separate_from_historical_fixed200_0ms",
                                  evaluated_clips=sum(r["evaluated_clips"] for r in by_delay.values()),
                                  normalized_return=sum(r["normalized_return"] for r in by_delay.values()) / len(by_delay),
                                  completed_clips=sum(r["completed_clips"] for r in by_delay.values()),
                                  joint_angle_error=joint_angle_summary(
                                      [row for stratum in by_delay.values() for row in stratum["clips"]], evaluator.mdp.robot_spec))
                elif controls is not None:
                    control_trace = trace.with_name(name + "_controls_trace.pt") if trace else None
                    result = evaluator.evaluate_paired(bundle["pi"], conditions, controls,
                                                       save_trace=trace, control_trace=control_trace)
                    if not result["valid_for_push_metrics"] or (name == "push120" and not qualified_push120(result)):
                        raise RuntimeError("Unverified planned suite cannot enter a queue result or the curriculum")
                else:
                    result = evaluator.evaluate(bundle["pi"], conditions=conditions, save_trace=trace)
                if name == "fixed200":
                    small_manifest = json.loads(Path(registry["fixed25"]["manifest"]).read_text())
                    small_names = set(small_manifest["clips"])
                    first = [row for row in result["clips"] if row["clip_id"] in small_names]
                    if len(first) != 25:
                        raise ValueError("Fixed200/fixed25 names disagree; cannot make a comparable quick baseline")
                    result["first25"] = dict(evaluated_clips=25, completed_clips=sum(r["completed"] for r in first),
                                             normalized_return=sum(r["normalized_return"] for r in first) / len(first),
                                             joint_angle_error=joint_angle_summary(first, evaluator.mdp.robot_spec))
                if name == "static16":
                    baseline = [c for c in result["clips"] if c.get("magnitude_mps") in (.5, 1.)]
                    result["baseline_8_all_no_native_termination_and_recovered_5s"] = len(baseline)==8 and all(c["completed"] and c["actual_injection"] and c["recovered_within_5s"] for c in baseline)
                    control_base = [c for c in result["no_push_controls"]["clips"] if c["condition_id"] in {b["no_push_control_id"] for b in baseline}]
                    result["baseline_8_controls_complete_and_stable_5s"] = len(control_base)==8 and all(c["completed"] and c["recovered_within_5s"] for c in control_base)
                results[name] = result
                write_json(run_root / "evaluations" / job["job_id"] / f"{name}.json", result)
            promoted = update_bests(queue, job, results)
            if promoted and "push120" not in results and job["reason"] not in ("divergence", "nonfinite_last_finite", "best_push", "best_candidate", "final"):
                queue.enqueue(job["actor"], step=job["vector_steps"], suites=["push120"], reason="best_push", priority=25)
            packet = dict(suites=results, promoted=promoted, retention_evaluated=True,
                          requested_suites=job["requested_suites"], evaluated_suites=list(results),
                          added_suites=job["added_suites"],
                          suite_statistics={name: dict(evaluated_clips=result.get("evaluated_clips", 0),
                              completed_clips=result.get("completed_clips", 0),
                              no_push_control_clips=result.get("no_push_controls", {}).get("evaluated_clips", 0))
                              for name,result in results.items()},
                          important_backup_pending=bool(promoted))
            if job["reason"] == "divergence":
                packet["quick_decision"] = quick_decision(results, job.get("baseline"))
            queue.finish(job, packet)
            intervention(run_root, "job_finished", job_id=job["job_id"], vector_step=job["vector_steps"], promoted=promoted)
        except BaseException as error:
            queue.fail(job, traceback.format_exc())
            intervention(run_root, "eval_failed", job_id=job["job_id"], error=repr(error))
        if once:
            return

