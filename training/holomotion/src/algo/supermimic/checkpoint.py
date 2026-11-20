# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Versioned learning-state recovery at an explicit physical episode boundary."""

import random
import os
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

# These affect execution placement, reporting or stopping, rather than updates.
RESUME_OVERRIDES = {
    "resume",
    "resume_compatibility_file",
    "output",
    "device",
    "max_hours",
    "max_vector_steps",
    "max_updates",
    "checkpoint_minutes",
    "policy_snapshot_minutes",
    "evaluate_minutes",
    "log_steps",
    "skip_evaluation",
    "evaluation_suites",
    "resume_interval_steps",
    "actor_snapshot_start",
    "actor_snapshot_interval",
    "full_evaluation_interval",
    "scalar_flush_records",
    "profile_actor_updates",
}
FORMAT = "supermimic_episode_boundary_v1"

# These modules manage logging, orchestration and read/write contracts. The
# learner, state/action representation, data sampling and physics equations are
# not eligible for an exemption. The paired graph-storage repair below has its
# own stricter GPU-evidence gate and is not part of this allowlist.
RESUME_COMPATIBLE_SOURCES = {
    "holomotion/src/algo/supermimic/retrain_runtime.py",
    "holomotion/src/algo/supermimic/retrain_eval_worker.py",
    "holomotion/src/algo/supermimic/checkpoint.py",
    "holomotion/src/algo/supermimic/config.py",
    "holomotion/src/algo/supermimic/scalar_logging.py",
    "holomotion/src/algo/supermimic/evaluation_queue.py",
}


GRAPH_STORAGE_SOURCES = {
    "holomotion/src/env/reference_tracking/warp_backend.py",
    "holomotion/src/env/reference_tracking/command_delay.py",
}


def graph_storage_compatibility(audit, bundle, specification, checkpoint, info, changes):
    """Opt in only to the paired graph-storage repair with independent GPU evidence.

    These physics-adjacent files are deliberately outside the orchestration
    allowlist. A missing, stale or differently scoped receipt always rejects.
    """
    approval = audit.get("graph_buffer_compatibility")
    graph_changes = changes & GRAPH_STORAGE_SOURCES
    if not graph_changes:
        if approval:
            raise ValueError("Graph storage audit lists a change that did not occur")
        return None
    if (graph_changes != GRAPH_STORAGE_SOURCES or not approval
            or approval.get("category") != "captured_graph_storage_lifetime_only"
            or not approval.get("receipt_path") or not approval.get("receipt_sha256")):
        raise ValueError("Graph storage source changes require the exact paired repair and independent GPU receipt")
    receipt_path = Path(approval["receipt_path"]).resolve()
    if receipt_path == Path(specification["arguments"]["resume_compatibility_file"]).resolve():
        raise ValueError("Graph storage evidence must be independent of the compatibility audit")
    payload = receipt_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != approval["receipt_sha256"]:
        raise ValueError("Graph storage GPU receipt hash changed")
    receipt = json.loads(payload)
    expected_parent = dict(path=str(checkpoint), vector_steps=bundle["schedule"]["vector_steps"],
                           bytes=info.st_size, mtime_ns=info.st_mtime_ns)
    if (receipt.get("format") != "pm01_011_graph_storage_equivalence_v1"
            or receipt.get("passed") is not True or receipt.get("checkpoint") != expected_parent):
        raise ValueError("Graph storage GPU evidence does not identify this immutable checkpoint")
    expected_sources = {name: {key: audit["allowed_source_changes"][name][key]
                              for key in ("before_sha256", "after_sha256")}
                        for name in graph_changes}
    if receipt.get("source_changes") != expected_sources:
        raise ValueError("Graph storage GPU receipt source hashes do not match the reviewed repair")
    saved = bundle["specification"]
    old_args, new_args = saved["arguments"], specification["arguments"]
    if (saved["physics"] != specification["physics"]
            or saved["runtime"].get("mujoco_warp_collision") != specification["runtime"].get("mujoco_warp_collision")
            or old_args.get("num_envs") != new_args.get("num_envs")
            or receipt.get("N") != new_args.get("num_envs")
            or old_args.get("capture_graph") is not True or new_args.get("capture_graph") is not True
            or receipt.get("capture_graph") is not True
            or old_args.get("amp") != "bf16" or new_args.get("amp") != "bf16"
            or receipt.get("training_amp") != "bf16" or receipt.get("dtype") != "float32"
            or not receipt.get("actual_gpu_uuid")
            or receipt["actual_gpu_uuid"] != specification["runtime"].get("cuda_visible_devices")
            or receipt.get("delay_ms") != [0, 5, 10, 15, 20]
            or not receipt.get("comparison_oracle")
            or type(receipt.get("max_abs_control_error")) not in (int, float)
            or receipt["max_abs_control_error"] != 0.0
            or receipt.get("unchanged") != dict(math=True, precision=True, mdp=True, training_recipe=True)):
        raise ValueError("Graph storage receipt changes or fails to verify the fixed physics/precision/recipe contract")
    for key in ("physical_graph_replays", "empty_cache_replays"):
        if type(receipt.get(key)) is not int or receipt[key] < 1:
            raise ValueError("Graph storage receipt lacks successful physical and allocator-stress graph replays")
    return dict(category=approval["category"], receipt_path=str(receipt_path),
                receipt_sha256=approval["receipt_sha256"], source_changes=expected_sources)


def resume_compatibility(bundle, specification):
    """Apply only an exact, checkpoint-bound audit; never mutate old weights."""
    arguments = specification["arguments"]
    path = arguments.get("resume_compatibility_file")
    if not path:
        return None
    audit = json.loads(Path(path).read_text())
    if audit.get("schema") != "pm01_011_resume_compatibility_v1" or audit.get("seed") != 13 or not audit.get("reason"):
        raise ValueError("Invalid 011 resume compatibility audit schema/seed/reason")
    checkpoint = Path(arguments["resume"]).resolve()
    info = checkpoint.stat()
    if (Path(audit.get("checkpoint_path", "")).resolve() != checkpoint
            or audit.get("checkpoint_vector_steps") != bundle["schedule"]["vector_steps"]
            or audit.get("checkpoint_size_bytes") != info.st_size
            or audit.get("checkpoint_mtime_ns") != info.st_mtime_ns):
        raise ValueError("Resume compatibility audit does not identify this immutable checkpoint")
    saved = bundle["specification"]
    if saved["arguments"].get("seed") != 13 or arguments.get("seed") != 13:
        raise ValueError("Compatibility audit cannot change the original seed")
    old, new = saved["source_sha256"], specification["source_sha256"]
    changes = {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
    allowed = audit.get("allowed_source_changes", {})
    eligible = RESUME_COMPATIBLE_SOURCES | GRAPH_STORAGE_SOURCES
    if set(allowed) != changes or not changes <= eligible:
        raise ValueError(f"Unreviewed or forbidden resume source changes: {sorted(changes ^ set(allowed) | (changes - eligible))}")
    for name, change in allowed.items():
        if (change.get("before_sha256") != old.get(name) or change.get("after_sha256") != new.get(name)
                or not change.get("reason") or old.get(name) is None or new.get(name) is None):
            raise ValueError(f"Resume source audit hash/reason mismatch: {name}")
    graph_storage = graph_storage_compatibility(audit, bundle, specification, checkpoint, info, changes)
    old_physics, new_physics = saved["physics"], specification["physics"]
    physics_changes = {key for key in old_physics.keys() | new_physics.keys() if old_physics.get(key) != new_physics.get(key)}
    approved_physics = audit.get("allowed_physics_changes", {})
    if set(approved_physics) != physics_changes or not physics_changes <= {"epa_horizon_capacity"}:
        raise ValueError("Compatibility audit may change only the approved EPA workspace capacity")
    if physics_changes:
        change = approved_physics["epa_horizon_capacity"]
        if (old_physics["epa_horizon_capacity"] != 24 or new_physics["epa_horizon_capacity"] != 64
                or change != {"before":24, "after":64}):
            raise ValueError("Only the verified EPA24 to historical EPA64 capacity restoration is allowed")
    old_collision = saved["runtime"].get("mujoco_warp_collision")
    new_collision = specification["runtime"].get("mujoco_warp_collision")
    if (physics_changes or old_collision != new_collision) and (not old_collision or not new_collision
            or old_collision.get("epa_horizon_capacity") != old_physics["epa_horizon_capacity"]
            or new_collision.get("epa_horizon_capacity") != new_physics["epa_horizon_capacity"]):
        raise ValueError("EPA physics/runtime capacity identities are inconsistent")
    approved_collision = audit.get("allowed_runtime_collision_change")
    if old_collision != new_collision:
        if (not approved_collision or approved_collision.get("before") != old_collision
                or approved_collision.get("after") != new_collision or not approved_collision.get("receipt_path")):
            raise ValueError("Runtime collision capacity change lacks an exact before/after audit and receipt")
        old_fixed = {k:v for k,v in old_collision.items() if k not in ("epa_horizon_capacity", "source_sha256")}
        new_fixed = {k:v for k,v in new_collision.items() if k not in ("epa_horizon_capacity", "source_sha256")}
        old_hash, new_hash = old_collision["source_sha256"], new_collision["source_sha256"]
        changed_hashes = {k for k in old_hash.keys() | new_hash.keys() if old_hash.get(k) != new_hash.get(k)}
        if (old_fixed != new_fixed or old_collision["epa_horizon_capacity"] != 24
                or new_collision["epa_horizon_capacity"] != 64 or changed_hashes != {"_src/types.py"}):
            raise ValueError("EPA compatibility cannot change collision algorithms or other runtime placement")
        if not Path(approved_collision["receipt_path"]).is_file():
            raise ValueError("Approved EPA restoration receipt is missing")
        receipt = json.loads(Path(approved_collision["receipt_path"]).read_text())
        historical = receipt.get("restored_historical_three_sha256", {})
        if (receipt.get("old_capacity") != 24 or receipt.get("restored_capacity") != 64
                or receipt.get("hard_overflow_guard_unchanged") is not True
                or receipt.get("private_environment") != new_collision.get("package_root")
                or receipt.get("original_types_sha256") != old_hash.get("_src/types.py")
                or any(historical.get(name) != new_hash.get("_src/" + name)
                       for name in ("types.py", "collision_gjk.py", "collision_convex.py"))):
            raise ValueError("EPA runtime hashes/capacity do not match the independent historical-restoration receipt")
    elif approved_collision:
        raise ValueError("Runtime collision audit lists a change that did not occur")
    return dict(audit_file=str(Path(path).resolve()), reason=audit["reason"],
                parent_checkpoint=dict(path=str(checkpoint), vector_steps=bundle["schedule"]["vector_steps"],
                                       size_bytes=info.st_size, mtime_ns=info.st_mtime_ns),
                source_changes=allowed, original_source_sha256=old, new_source_sha256=new,
                physics_changes=approved_physics, runtime_collision_change=approved_collision,
                graph_storage=graph_storage,
                old_checkpoint_modified=False)


def validate_checkpoint(bundle, specification):
    """Reject inference bundles and incompatible data, sources or recipes."""
    if bundle.get("format") != FORMAT:
        raise ValueError(
            "Checkpoint lacks complete episode-boundary learning state"
        )
    saved = bundle["specification"]
    compatibility = resume_compatibility(bundle, specification)
    before, after = saved["arguments"], specification["arguments"]
    keys = (before.keys() | after.keys()) - RESUME_OVERRIDES
    changed = sorted(key for key in keys if before.get(key) != after.get(key))
    if changed:
        raise ValueError(f"Resume recipe changed: {changed}")
    for field in (
        "source_sha256",
        "input_sha256",
        "validation",
        "generated_model_sha256",
        "schedule",
        "parameters",
        "actor_observation_dim",
        "goal_dim",
        "physics",
    ):
        if saved[field] != specification[field] and not (compatibility and field in ("source_sha256", "physics")):
            raise ValueError(f"Resume contract changed: {field}")
    if saved.get("input_shard_identity") != specification.get("input_shard_identity"):
        raise ValueError("Resume HDF5 shard size/mtime identity changed")
    for field in ("torch", "mujoco", "warp", "mujoco_warp", "torch_cuda"):
        if saved["runtime"][field] != specification["runtime"][field]:
            raise ValueError(f"Resume runtime changed: {field}")
    if compatibility:
        specification["resume_compatibility"] = compatibility


def capture_learning_state(learner, buffer, schedule, env):
    """Keep replay on CPU and leave simulator internals outside this contract."""
    return dict(
        format=FORMAT,
        exact_training_resume=False,
        resume_semantics="Restore learning state; reset all physical episodes and observation histories",
        **{
            name: cpu_snapshot(getattr(learner, name).state_dict())
            for name in ("wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v")
        },
        schedule=schedule.state_dict(),
        replay=buffer.state_dict(),
        curriculum={
            name: getattr(env, name).to("cpu", copy=True)
            for name in ("bin_G", "bin_w_avg")
        },
        torch_rng=torch.random.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(learner.dev),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        stronger_reset_fraction=float(getattr(env, "stronger_reset_fraction", 0.0)),
        source_sampler_counts=cpu_snapshot({name: getattr(env.source_sampler, name)
                                            for name in ("reset_counts", "transition_counts")})
        if getattr(env, "source_sampler", None) is not None else None,
        coverage_counts=cpu_snapshot(getattr(learner, "coverage_counts", {})),
    )


def restore_learning_state(bundle, learner, buffer, schedule, env):
    """Restore RNG after initialization, then draw fresh physical resets."""
    # Validate schedule/replay before mutating model weights.
    schedule.load_state_dict(bundle["schedule"])
    buffer.load_state_dict(bundle["replay"])
    for name in ("bin_G", "bin_w_avg"):
        source, target = bundle["curriculum"][name], getattr(env, name)
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError("Curriculum shape or dtype changed")
        if not torch.isfinite(source).all():
            raise ValueError("Nonfinite curriculum")
        target.copy_(source)
    for name in ("wm", "pi", "critic"):
        getattr(learner, name).load_state_dict(bundle[name], strict=True)
    for name in ("opt_wm", "opt_pi", "opt_v"):
        getattr(learner, name).load_state_dict(bundle[name])
    torch.random.set_rng_state(bundle["torch_rng"])
    torch.cuda.set_rng_state(bundle["cuda_rng"], learner.dev)
    random.setstate(bundle["python_rng"])
    np.random.set_state(bundle["numpy_rng"])
    buffer.mark_episode_boundary()
    if hasattr(env, "stronger_reset_fraction"):
        env.stronger_reset_fraction = float(bundle.get("stronger_reset_fraction", 0.0))
    if bundle.get("source_sampler_counts") and getattr(env, "source_sampler", None) is not None:
        for name, source in bundle["source_sampler_counts"].items():
            getattr(env.source_sampler, name).copy_(source)
    for name, source in bundle.get("coverage_counts", {}).items():
        if name in getattr(learner, "coverage_counts", {}):
            learner.coverage_counts[name].copy_(source)
    env.reset(torch.arange(env.N, device=learner.dev))


def cpu_snapshot(value):
    """Own immutable CPU storage before handing a snapshot to an I/O thread."""
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def atomic_torch_save(bundle, path, *, verify_recovery=False):
    """The prior latest checkpoint remains intact until the new file is durable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as stream:
        torch.save(bundle, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if verify_recovery:
        # mmap reads the zip/pickle structure and tiny counters without
        # repeatedly reading multi-GB replay tensors or recalculating SHA.
        loaded = torch.load(temporary, map_location="cpu", mmap=True, weights_only=False)
        if set(loaded) != set(bundle) or loaded.get("schedule") != bundle.get("schedule"):
            raise ValueError("Written recovery state failed metadata/counter read-back")
        if loaded.get("format") != bundle.get("format"):
            raise ValueError("Written recovery format changed")
        del loaded
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class RecoveryWriter:
    """One outstanding complete state, avoiding stale unlimited checkpoint copies."""
    def __init__(self, output):
        self.output = Path(output)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="011-recovery-writer")
        self.pending = None

    def busy(self):
        if self.pending is None:
            return False
        if self.pending.done():
            self.pending.result()
            self.pending = None
            return False
        return True

    def submit(self, bundle, *, milestone=False):
        if self.busy():
            return False
        step = bundle["schedule"]["vector_steps"]
        self.pending = self.pool.submit(commit_recovery, bundle, self.output, milestone=milestone)
        return True

    def close(self):
        self.pool.shutdown(wait=True)
        if self.pending is not None:
            self.pending.result()


def commit_recovery(bundle, output, *, milestone=False):
    """One latest plus one prior finite guard; independent tags retain their files.

    Approved 2026-11-16: untagged intermediate rolling points may be superseded
    by a verified newer complete point without a Drive upload for each one.
    """
    from .evaluation_queue import write_json
    import json
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    latest = output / "latest_resume.pt"
    prior = output / "previous_unconfirmed_resume.pt"
    prior_record = output / "previous_unconfirmed_resume.json"
    if latest.exists():
        # Hard link keeps the old inode through os.replace without another
        # multi-GB copy. This is one bounded guard, never an unbounded history.
        new_prior = prior.with_suffix(".pt.next")
        new_prior.unlink(missing_ok=True)
        os.link(latest, new_prior)
        os.replace(new_prior, prior)
        record = output / "latest_resume.json"
        previous_step = json.loads(record.read_text())["vector_steps"] if record.exists() else -1
        write_json(prior_record, dict(path=str(prior), vector_steps=previous_step,
                                      purpose="one_previous_verified_finite_guard; bounded_rolling_not_independent_backup_tag"))
    atomic_torch_save(bundle, latest, verify_recovery=True)
    if milestone:
        milestone_path = output / "resume_0000010000.pt"
        if not milestone_path.exists():
            # A stable independent name for the approved 10k full checkpoint.
            os.link(latest, milestone_path)
    step = bundle["schedule"]["vector_steps"]
    write_json(output / "latest_resume.json", dict(
        path=str(latest), vector_steps=step, bytes=latest.stat().st_size,
        world_updates=bundle["schedule"]["world_updates"], actor_updates=bundle["schedule"]["actor_updates"],
        previous_unconfirmed_guard=str(prior) if prior.exists() else None,
        readback_verified=True, important_backup_required=True,
        semantics="Complete learning state; physical episodes and delay queues reset on restore"))
