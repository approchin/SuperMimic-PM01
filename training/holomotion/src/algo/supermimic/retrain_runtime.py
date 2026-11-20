# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""PM01 011 single-run collection, complete logs and independent evaluation."""

import json
import math
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .evaluation_queue import EvaluationQueue, write_json
from .scalar_logging import ScalarLogger, json_safe


def intervention(output, kind, **detail):
    path = Path(output) / "logs/intervention_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    event = dict(event=kind, utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), **detail)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o664)
    try:
        os.write(descriptor, (json.dumps(json_safe(event), allow_nan=False) + "\n").encode())
        if kind in ("stopped", "nonfinite", "divergence_suspected"):
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return event


def publish_cpu_fault_status(output, *, error, reason, vector_steps, world_updates,
                             actor_updates, training_wall_s, cleanup_errors=None):
    """Publish a failure without touching CUDA or device-backed telemetry.

    A poisoned context cannot supply coverage, memory or pending scalar values.
    Retained telemetry is explicitly attributed to the last published status.
    """
    path = Path(output) / "status.json"
    try:
        previous = json.loads(path.read_text())
    except (OSError, ValueError):
        previous = {}
    reused = [name for name in (
        "current_metrics", "coverage_counts", "reset_counts",
        "transition_counts_by_group", "memory", "transitions",
        "F_optimizer_steps", "critic_optimizer_steps", "eval_queue",
        "stronger_reset_fraction", "checkpoint_seconds") if name in previous]
    origin = previous.get("last_good_status") if previous.get("status_publication") == "cpu_only_fault_status_v1" else None
    if origin is None:
        origin = dict(path=str(path), vector_steps=previous.get("vector_steps"),
                      last_update_utc=previous.get("last_update_utc"), pid=previous.get("pid"),
                      reused_fields=reused)
    now = datetime.now(timezone.utc).isoformat()
    value = dict(previous, phase="failed", reason=reason, error=repr(error),
                 pid=os.getpid(), seed=13, vector_step=vector_steps, vector_steps=vector_steps,
                 world_events=world_updates, actor_updates=actor_updates,
                 last_update_utc=now, training_wall_s=training_wall_s,
                 status_publication="cpu_only_fault_status_v1", last_good_status=origin,
                 telemetry_current_at_fault=False,
                 fault=dict(utc=now, pid=os.getpid(), vector_steps=vector_steps,
                            exception_type=type(error).__name__, error=repr(error)),
                 cleanup_errors=dict(cleanup_errors or {}))
    write_json(path, json_safe(value))
    return value


def learning_state_finiteness(learner):
    """Check checkpointed networks/Adam state; simulator state is a separate domain."""
    import torch
    tensors, bad_scalars = {}, []
    def visit(value, name):
        if isinstance(value, torch.Tensor):
            if value.is_floating_point() or value.is_complex():
                tensors.setdefault(str(value.device), []).append((name, value))
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{name}.{key}")
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                visit(item, f"{name}.{index}")
        elif isinstance(value, float) and not math.isfinite(value):
            bad_scalars.append(name)
    for name in ("pi", "wm", "critic", "opt_pi", "opt_wm", "opt_v"):
        visit(getattr(learner, name).state_dict(), name)
    bad = list(bad_scalars)
    checked = 0
    for items in tensors.values():
        flags = torch.stack([torch.isfinite(value).all() for _,value in items]).cpu().tolist()
        checked += len(flags)
        bad.extend(name for (name,_), finite in zip(items, flags) if not finite)
    return dict(finite=not bad, checked_floating_tensors=checked, nonfinite_fields=bad[:32])


def preserve_rejected_physics_state(*, learner, schedule, recovery, capture_state,
                                    actor_snapshot, recovery_output):
    """Save the last accepted learning boundary after a rejected physical step.

    The failed transition is absent from replay. Simulator positions/queues are
    deliberately outside episode-boundary recovery and will reset on restore.
    All failures leave the previously published resume file available.
    """
    from .checkpoint import commit_recovery
    report = dict(vector_steps=schedule.vector_steps, failed_transition_entered_replay=False,
                  actor_saved=False, recovery_saved=False,
                  resume_semantics="Restore finite learning state; reset all physical episodes/history/delay queues")
    try:
        report["learning_state"] = learning_state_finiteness(learner)
    except BaseException as error:
        report.update(reason="learning_finite_check_failed", error=repr(error))
        return report
    if not report["learning_state"]["finite"]:
        report["reason"] = "learning_state_nonfinite_keep_prior_resume"
        return report
    if getattr(schedule, "pending_world", 0) or getattr(schedule, "pending_actor", 0):
        report["reason"] = "optimizer_event_in_progress_keep_prior_quiescent_resume"
        return report
    tag = f"physical_fault_last_finite_{time.time_ns()}"
    try:
        report["actor"] = actor_snapshot(tag, fault=True)
        report["actor_saved"] = True
    except BaseException as error:
        report["actor_save_error"] = repr(error)
    try:
        recovery.close()  # Finish an older async write before publishing this newer boundary.
    except BaseException as error:
        report["older_async_writer_error"] = repr(error)
    try:
        bundle = capture_state()
        recovery_output = Path(recovery_output)
        commit_recovery(bundle, recovery_output)
        report.update(recovery_saved=True, latest_resume=str(recovery_output / "latest_resume.pt"))
        # Independent fault tag has an important backup obligation; rolling
        # latest/guard replacement cannot remove the underlying inode.
        fault = recovery_output / f"fault_resume_{schedule.vector_steps:010d}_{tag}.pt"
        os.link(recovery_output / "latest_resume.pt", fault)
        report.update(fault_resume=str(fault), important_backup_required=True)
        report["reason"] = "rejected_physical_transition_finite_learning_boundary_preserved"
    except BaseException as error:
        report.setdefault("reason", "finite_learning_saved_but_fault_tag_failed" if report["recovery_saved"] else "fault_recovery_save_failed_keep_prior_resume")
        report["recovery_save_error"] = repr(error)
    return report


def run_retrain(args, *, env, model, motion, mdp, specification, pressure, launched):
    import torch
    from .buffer import ReplayBuffer
    from .checkpoint import (RecoveryWriter, atomic_torch_save, capture_learning_state, commit_recovery,
                             cpu_snapshot, restore_learning_state, validate_checkpoint)
    from .learner import ReferenceLearner
    from .monitoring import DivergenceMonitor, EpisodeMonitor, PushRecoveryCurriculum
    from .schedule import ScheduleConfig, UpdateSchedule
    from holomotion.src.env.reference_tracking.mdp import normalize_xi, unflatten_state

    if args.sampling_index is None:
        raise ValueError("011 requires the verified 3982 sampling_index before first collection")
    env.configure_sampling(args.sampling_index)  # Resets every lane before history/replay exists.
    output = Path(args.output)
    for name in ("logs", "checkpoints/actors", "checkpoints/recovery"):
        (output / name).mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        with torch.no_grad():
            transition = None
            for _ in range(max(args.warmup_steps, 1)):
                transition = env.step(torch.zeros(args.num_envs, mdp.robot_spec.actions, device=args.device))
        atomic_torch_save(cpu_snapshot(transition), output / "physical_preflight_last_transition.pt")
        return dict(scope="Physical execution only; no learner or optimizer constructed", N=args.num_envs,
                    physics_status=env.status_summary(), specification=specification)
    logger = ScalarLogger(output / "logs/scalars.jsonl", flush_records=args.scalar_flush_records)
    queue = EvaluationQueue(output / "eval_queue")
    recovery = RecoveryWriter(output / "checkpoints/recovery")
    buffer = ReplayBuffer(args.buffer_steps, args.num_envs, args.device, robot_spec=mdp.robot_spec)
    learner = ReferenceLearner(args, motion, mdp, buffer)
    learner.scalar_logger = logger
    learner.coverage_counts = {name: torch.zeros(3, dtype=torch.long, device=args.device)
                               for name in ("F_windows", "F_valid_targets", "actor_starts", "actor_valid_predictions")}
    original_windows = buffer.sample_windows
    def counted_windows(*values, **keywords):
        window = original_windows(*values, **keywords)
        groups = env.source_sampler.clip_group[motion.clip_id[window["f"]]]
        learner.coverage_counts["F_windows"].scatter_add_(0, groups[:, 0], torch.ones_like(groups[:, 0]))
        valid_groups = groups[window["mask"].bool()]
        learner.coverage_counts["F_valid_targets"].scatter_add_(0, valid_groups, torch.ones_like(valid_groups))
        return window
    buffer.sample_windows = counted_windows
    original_starts = learner.sample_valid_starts
    def counted_starts(*values, **keywords):
        start, histories = original_starts(*values, **keywords)
        groups = env.source_sampler.clip_group[motion.clip_id[start["f"]]]
        learner.coverage_counts["actor_starts"].scatter_add_(0, groups, torch.ones_like(groups))
        valid = (motion.clip_end[motion.clip_id[start["f"]]] - start["f"]).clamp(0, args.horizon)
        learner.coverage_counts["actor_valid_predictions"].scatter_add_(0, groups, valid)
        return start, histories
    learner.sample_valid_starts = counted_starts
    schedule = UpdateSchedule(ScheduleConfig(
        physical_envs=args.num_envs, actor_batch_size=args.pi_batch,
        imagination_horizon=args.horizon, world_batch_size=args.wm_batch,
        world_window_length=args.wm_window, world_batches_per_update=args.wm_batches,
        actor_updates_per_vector_step=args.utd, world_updates_per_vector_step=args.utd,
        warmup_vector_steps=args.warmup_steps, world_only_updates=args.world_only_updates,
    ))
    specification.update(schedule=asdict(schedule.config), steady_sample_ratios=schedule.ratios(),
                         parameters={name: sum(p.numel() for p in getattr(learner, name).parameters())
                                     for name in ("wm", "pi", "critic")},
                         run_id="retrain_011_seed13_20261116", seed=13,
                         sampler=dict(index=str(args.sampling_index), ordinary_run_jump=[.9, .05, .05],
                                      source_bin_mixture=[.5, .4, .1], source_variant_share_preserved=True),
                         divergence_protocol=dict(window_actual_updates=args.divergence_window,
                             consecutive_windows=args.divergence_windows, factor=args.divergence_factor,
                             healthy_baseline="median of first three window medians, then latest ten healthy windows",
                             absolute_floor=1e-6, actor_sign_ignored=True,
                             action="pause_updates_then_priority0_real_quick25_and_target18",
                             quick_completion_drop_count=2, quick_return_drop=0.05,
                             quick_timeout_s=args.quick_check_timeout_s, nonfinite="immediate_stop"))
    monitor = EpisodeMonitor()
    curriculum = PushRecoveryCurriculum()
    previous_wall = 0.0
    if args.resume is not None:
        bundle = torch.load(args.resume, map_location="cpu", weights_only=False)
        validate_checkpoint(bundle, specification)
        if specification.get("resume_compatibility"):
            intervention(output, "resume_compatibility_applied", **specification["resume_compatibility"])
        restore_learning_state(bundle, learner, buffer, schedule, env)
        previous_wall = bundle.get("training_wall_s", 0.0)
        vars(monitor).update(bundle.get("monitor", {}))
        curriculum = PushRecoveryCurriculum(bundle.get("push_curriculum"))
        env.stronger_reset_fraction = curriculum.fraction
        del bundle
    write_json(output / "specification.json", specification)
    divergence = DivergenceMonitor(args.divergence_window, args.divergence_windows, args.divergence_factor)
    history = torch.zeros(args.num_envs, args.hist, mdp.obs_dim, device=args.device)
    latest_actor = None
    latest_actor_step = schedule.vector_steps
    last_recovery_step = schedule.vector_steps
    last_status = 0.0
    latest_metrics = {}
    seen_results, latest_comparison = set(), None
    stop_reason = None
    phase = "profiling" if args.profile_actor_updates else "training"
    optimizer_counts = dict(F=schedule.world_updates * args.wm_batches,
                            Critic=schedule.actor_updates * args.critic_epochs * args.critic_minibatches)
    checkpoint_seconds = 0.0
    failure = None
    fault_cleanup_errors = {}

    def wall():
        return previous_wall + time.monotonic() - launched

    def status(current_phase=None, reason=None):
        nonlocal last_status
        value = dict(phase=current_phase or phase, pid=os.getpid(), seed=13,
                     vector_step=schedule.vector_steps, vector_steps=schedule.vector_steps,
                     N=args.num_envs, transitions=monitor.transitions,
                     world_events=schedule.world_updates, F_optimizer_steps=optimizer_counts["F"],
                     actor_updates=schedule.actor_updates,
                     critic_optimizer_steps=optimizer_counts["Critic"] + logger.module_counts["critic"],
                     last_update_utc=datetime.now(timezone.utc).isoformat(), training_wall_s=wall(),
                     latest_actor=latest_actor, latest_resume=str(output / "checkpoints/recovery/latest_resume.pt"),
                     eval_queue={name: len(list((queue.root / name).glob("*.json")))
                                 for name in ("pending", "running", "completed", "failed")},
                     current_metrics=latest_metrics, stronger_reset_fraction=curriculum.fraction,
                     reason=reason, checkpoint_seconds=checkpoint_seconds,
                     coverage_counts={name: tensor.tolist() for name,tensor in learner.coverage_counts.items()},
                     reset_counts=env.source_sampler.reset_counts.tolist(),
                     transition_counts_by_group=env.source_sampler.transition_counts.tolist(),
                     memory=dict(allocated=torch.cuda.memory_allocated(args.device),
                                 reserved=torch.cuda.memory_reserved(args.device),
                                 peak_allocated=torch.cuda.max_memory_allocated(args.device)))
        write_json(output / "status.json", json_safe(value))
        last_status = time.monotonic()

    def actor_snapshot(tag=None, *, fault=False):
        nonlocal latest_actor, latest_actor_step
        suffix = f"_{tag}" if tag else ""
        path = output / "checkpoints/actors" / f"actor_{schedule.vector_steps:010d}{suffix}.pt"
        atomic_torch_save(dict(format="supermimic_policy_progress_v1", resume_supported=False,
                              pi=cpu_snapshot(learner.pi.state_dict()), specification=specification,
                              schedule=schedule.state_dict(), training_wall_s=wall()), path)
        latest_actor = str(path)
        latest_actor_step = schedule.vector_steps
        queue.assign(path, "latest_actor")
        if fault:
            queue.protect(path, "key_fault")
        return str(path)

    def complete_state():
        bundle = capture_learning_state(learner, buffer, schedule, env)
        bundle.update(specification=specification, training_wall_s=wall(), monitor=vars(monitor).copy(),
                      push_curriculum=curriculum.state_dict(), training_reward_points=[])
        return bundle

    def collect_results():
        nonlocal latest_comparison
        for path in sorted((queue.root / "results").glob("*.json")):
            if path.name in seen_results:
                continue
            result = json.loads(path.read_text())
            seen_results.add(path.name)
            suites = result.get("suites", {})
            quick = suites.get("fixed25") or suites.get("fixed200", {}).get("first25")
            if (quick and result.get("reason") != "divergence"
                    and (latest_comparison is None or result["vector_steps"] >= latest_comparison["vector_steps"])):
                latest_comparison = dict(fixed25=quick, target18=suites.get("target18", {}),
                                         actor=result["actor"], vector_steps=result["vector_steps"])
                write_json(output / "latest_real_assessment.json", latest_comparison)
            change = curriculum.observe(result, current_step=schedule.vector_steps, diverging=divergence.alerted)
            if change:
                env.stronger_reset_fraction = curriculum.fraction
                intervention(output, "reset_curriculum_changed", **change)
                write_json(output / "push_curriculum.json", curriculum.state_dict())

    def check_divergence():
        logger.flush()
        events = logger.take_events()
        for event in events:
            if event["module"] in ("actor", "critic_summary", "world_summary", "collection"):
                latest_metrics[event["module"]] = event["metrics"]
        alarms = divergence.observe(events)
        if not alarms:
            return None
        actor = actor_snapshot("divergence", fault=True)
        event = intervention(output, "divergence_suspected", vector_step=schedule.vector_steps, alarms=alarms, actor=actor)
        job = queue.enqueue(actor, step=schedule.vector_steps, suites=["fixed25", "target18"],
                            reason="divergence", priority=0, baseline=latest_comparison)
        status("paused_for_fast_real_inference")
        deadline = time.monotonic() + args.quick_check_timeout_s
        result_path = queue.root / "results" / f"{job['job_id']}.json"
        while time.monotonic() < deadline:
            if result_path.exists():
                result = json.loads(result_path.read_text())
                decision = result.get("quick_decision", {})
                intervention(output, "fast_real_inference_complete", vector_step=schedule.vector_steps, decision=decision)
                if decision.get("stop", True):
                    return decision.get("reason", "fast_real_inference_unresolved_keep_updates_stopped")
                divergence.alerted = False
                divergence.bad.clear()
                status("training")
                return None
            if (queue.root / "failed" / f"{job['job_id']}.json").exists():
                intervention(output, "eval_failed", job_id=job["job_id"])
                return "divergence_quick_evaluation_failed"
            time.sleep(0.25)
        return "divergence_quick_evaluation_timeout_updates_remain_stopped"

    try:
        intervention(output, "run_started", profile=bool(args.profile_actor_updates), N=args.num_envs,
                     utd_events_per_vector_step=args.utd, resume=str(args.resume) if args.resume else None)
        status()
        if not args.profile_actor_updates and args.resume is None:
            initial_actor = actor_snapshot("initial")
            queue.enqueue(initial_actor, step=schedule.vector_steps, suites=["fixed25", "target18"],
                          reason="initial", priority=50)
        while True:
            if args.max_vector_steps and schedule.vector_steps >= args.max_vector_steps:
                stop_reason = "explicit_vector_step_bound"; break
            if args.max_updates and schedule.world_updates >= args.max_updates:
                stop_reason = "explicit_update_bound"; break
            if args.max_hours and wall() >= args.max_hours * 3600:
                stop_reason = "explicit_time_bound"; break
            control = output / "control.json"
            if control.exists() and json.loads(control.read_text()).get("stop"):
                stop_reason = json.loads(control.read_text()).get("reason", "requested_stop"); break
            with torch.no_grad():
                observation = mdp.add_noise(mdp.obs(env.state(), env.a_prev, env.xi, env.f))
                history = torch.where((env.ep_step == 0)[:, None, None], observation[:, None].expand_as(history),
                                      torch.cat((history[:, 1:], observation[:, None]), 1))
                goal = motion.goal(env.f, args.goal_frames)
                learner.pi.obs_norm.update(observation); learner.pi.goal_norm.update(goal)
                action = (learner.pi(history, goal) + args.action_noise * torch.randn(args.num_envs, mdp.robot_spec.actions, device=args.device)).clamp(-20, 20)
                pressure.before_launch()
                transition = env.step(action)
                monitor.add(transition); buffer.add(transition)
                state, following = unflatten_state(transition["s"]), unflatten_state(transition["s1"])
                learner.wm.in_norm.update(learner.wm.inputs(state, transition["a"], normalize_xi(transition["xi"]), transition.get("delay_state")))
                learner.wm.out_norm.update(learner.wm.targets(state, following))
            worlds, actors = schedule.advance()
            lr = args.lr * (.03 + .97 * .5 * (1 + math.cos(math.pi * min(schedule.world_updates / args.lr_decay_updates, 1.0))))
            learner.set_learning_rate(lr)
            logger.set_context(vector_step=schedule.vector_steps, transitions=monitor.transitions, N=args.num_envs,
                               world_event=schedule.world_updates, actor_update=schedule.actor_updates, lr=lr)
            logger.emit("collection", dict(real_mean_step_reward=transition["r"].mean()))
            for _ in range(worlds):
                losses, errors, gradients = [], [], []
                for batch in range(args.wm_batches):
                    loss, error = learner.wm_update()
                    optimizer_counts["F"] += 1
                    grad = learner.last_world_gradient_norm
                    logger.emit("world", dict(loss=loss, last_step_error=error, gradient_norm=grad),
                                update_index=optimizer_counts["F"], batch_index=batch)
                    losses.append(loss); errors.append(error); gradients.append(grad)
                logger.emit("world_summary", dict(loss_mean=torch.stack(losses).mean(), loss_max=torch.stack(losses).max(),
                                                    gradient_norm_mean=torch.stack(gradients).mean(), gradient_norm_max=torch.stack(gradients).max()))
                schedule.complete_world()
            for _ in range(actors):
                logger.set_context(actor_update=schedule.actor_updates + 1)
                learner.policy_update()  # Actor, Jreward/Jtail and every Critic minibatch emit through the callback.
                schedule.complete_actor()
                if schedule.actor_updates == 1:
                    intervention(output, "first_actor_critic_update_complete", vector_step=schedule.vector_steps, N=args.num_envs)
            reason = check_divergence()
            if reason:
                stop_reason = reason; break
            step = schedule.vector_steps
            if args.profile_actor_updates:
                if schedule.actor_updates >= args.profile_actor_updates:
                    stop_reason = "complete_update_profile_finished"; break
            else:
                if step % args.full_evaluation_interval == 0:
                    actor = actor_snapshot()
                    queue.enqueue(actor, step=step, suites=["fixed200", "target18", "push120"], reason="full", priority=20)
                elif step > args.actor_snapshot_start and step % args.actor_snapshot_interval == 0:
                    actor = actor_snapshot()
                    queue.enqueue(actor, step=step, suites=["fixed200", "target18"], reason="dense", priority=40)
                elif step % 5000 == 0:
                    actor = actor_snapshot()
                    queue.enqueue(actor, step=step, suites=["fixed25", "target18"], reason="small", priority=30)
                milestone = step == 10000
                if milestone or step - last_recovery_step >= args.resume_interval_steps:
                    if milestone and recovery.pending is not None:
                        recovery.pending.result()
                    if not recovery.busy():
                        before = time.monotonic()
                        recovery.submit(complete_state(), milestone=milestone)
                        checkpoint_seconds += time.monotonic() - before
                        last_recovery_step = step
                if step % args.log_steps == 0:
                    collect_results(); queue.prune_evaluated()
            if step % args.log_steps == 0 or time.monotonic() - last_status >= 15:
                metrics = {f"{name}_{group}":tensor[i] for name,tensor in learner.coverage_counts.items()
                           for i,group in enumerate(env.source_sampler.groups)}
                metrics.update({f"resets_{group}":env.source_sampler.reset_counts[i]
                                for i,group in enumerate(env.source_sampler.groups)})
                metrics.update({f"transitions_{group}":env.source_sampler.transition_counts[i]
                                for i,group in enumerate(env.source_sampler.groups)})
                logger.emit("sampling_coverage", metrics)
                status()
                print(json.dumps(dict(phase=phase, vector_step=step, actor_updates=schedule.actor_updates,
                                      world_events=schedule.world_updates, transitions=monitor.transitions)), flush=True)
    except BaseException as error:
        failure = error
        stop_reason = "nonfinite" if isinstance(error, FloatingPointError) or "nonfinite" in str(error).lower().replace("-", "") else "runtime_fault"
        # Persist the original cause before any best-effort device diagnostic,
        # checkpoint or logger cleanup can encounter the poisoned context.
        def publish_fault():
            return publish_cpu_fault_status(
                output, error=failure, reason=stop_reason, vector_steps=schedule.vector_steps,
                world_updates=schedule.world_updates, actor_updates=schedule.actor_updates,
                training_wall_s=wall(), cleanup_errors=fault_cleanup_errors)
        try:
            publish_fault()
        except BaseException as status_error:
            fault_cleanup_errors["initial_cpu_status"] = repr(status_error)
        diagnostic = getattr(error, "physics_diagnostic", None)
        rejected_physics = diagnostic is not None and diagnostic.get("failed_transition_entered_replay") is False
        fault_state = None
        try:
            if rejected_physics and not args.profile_actor_updates:
                fault_state = preserve_rejected_physics_state(
                    learner=learner, schedule=schedule, recovery=recovery, capture_state=complete_state,
                    actor_snapshot=actor_snapshot, recovery_output=output / "checkpoints/recovery")
                write_json(output / "logs/fault_learning_state.json", fault_state)
            intervention(output, stop_reason, vector_step=schedule.vector_steps, error=repr(error),
                         latest_finite_actor=latest_actor, failed_transition_entered_replay=False if rejected_physics else None,
                         fault_learning_state=fault_state)
            if diagnostic is not None:
                try:
                    atomic_torch_save(cpu_snapshot(diagnostic), output / "logs/physics_fault.pt")
                except BaseException as diagnostic_error:
                    fault_cleanup_errors["physics_diagnostic_save"] = repr(diagnostic_error)
            if latest_actor:
                queue.protect(latest_actor, "key_fault_last_known_snapshot")
                reason = ("physical_fault_last_finite" if fault_state and fault_state.get("actor_saved") else
                          "physical_fault_older_snapshot" if rejected_physics else "nonfinite_last_finite")
                queue.enqueue(latest_actor, step=latest_actor_step, suites=["fixed25", "target18"], reason=reason, priority=0)
        except BaseException as diagnostic_error:
            fault_cleanup_errors["fault_diagnostics"] = repr(diagnostic_error)
        raise
    finally:
        if failure is not None:
            for name, close in (("logger_close", logger.close), ("recovery_close", recovery.close)):
                try:
                    close()
                except BaseException as cleanup_error:
                    fault_cleanup_errors[name] = repr(cleanup_error)
            try:
                intervention(output, "stopped", vector_step=schedule.vector_steps, reason=stop_reason,
                             error=repr(failure), cleanup_errors=fault_cleanup_errors)
            except BaseException as cleanup_error:
                fault_cleanup_errors["stopped_event"] = repr(cleanup_error)
            try:
                publish_fault()
            except BaseException as cleanup_error:
                # Never replace the original exception with a cleanup failure.
                print(f"CPU fault status write failed: {cleanup_error!r}; original: {failure!r}", flush=True)
        else:
            logger.close()
            recovery.close()
            intervention(output, "stopped", vector_step=schedule.vector_steps, reason=stop_reason)
            status("profile_finished" if args.profile_actor_updates else "stopped", stop_reason)
    if not args.profile_actor_updates and stop_reason not in ("nonfinite", "runtime_fault"):
        commit_recovery(complete_state(), output / "checkpoints/recovery")
    return dict(scope="PM01011 single seed13 approved training", stop_reason=stop_reason,
                clock=schedule.state_dict(), sample_counts=schedule.ratios(), N=args.num_envs,
                collected_real_transitions=monitor.transitions, training_wall_s=wall(),
                peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                physics_status=env.status_summary(), specification=specification)

