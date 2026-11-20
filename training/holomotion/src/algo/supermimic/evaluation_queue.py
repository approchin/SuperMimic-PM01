# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Shared-disk evaluation jobs, atomic worker claims and protected retention."""

import json
import os
import time
import uuid
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def evaluation_summary(packet):
    """Bounded status for hourly inspection; detailed evidence stays in results."""
    summary = {key: packet[key] for key in (
        'job_id', 'actor', 'vector_steps', 'reason', 'finished_at', 'promoted',
        'quick_decision', 'retention_evaluated', 'important_backup_pending',
        'requested_suites', 'evaluated_suites', 'added_suites') if key in packet}
    keep = (
        'evaluated_clips', 'completed_clips', 'normalized_return', 'groups', 'first25', 'joint_angle_error',
        'valid_for_push_metrics', 'invalid_reason', 'reason', 'planned_conditions',
        'condition_events_reached', 'actual_injections', 'pre_injection_failures',
        'perturbation_route', 'shared_prefix_verified_pairs', 'control_pairing_scope',
        'light_completed', 'medium_completed', 'light_denominator', 'medium_denominator',
        'baseline_8_all_no_native_termination_and_recovered_5s',
        'baseline_8_controls_complete_and_stable_5s')
    summary['suites'] = {}
    for name, suite in packet.get('suites', {}).items():
        small = {key: suite[key] for key in keep if key in suite}
        control = suite.get('no_push_controls')
        if control is not None:
            small['no_push_controls'] = {key: control[key] for key in (
                'evaluated_clips', 'completed_clips', 'actual_injections',
                'condition_events_reached', 'normalized_return', 'joint_angle_error') if key in control}
        if 'by_delay_ms' in suite:
            small['by_delay_ms'] = {delay: {key: value[key] for key in (
                'evaluated_clips', 'completed_clips', 'normalized_return', 'joint_angle_error') if key in value}
                for delay, value in suite['by_delay_ms'].items()}
        summary['suites'][name] = small
    summary['format'] = 'pm01_011_small_evaluation_summary_v2'
    return summary


def serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.locked():
            return method(self, *args, **kwargs)
    return wrapped


class EvaluationQueue:
    def __init__(self, root):
        self.root = Path(root)
        for name in ("pending", "running", "completed", "failed", "results", "protection", "summaries"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self._thread_lock = threading.RLock()
        self._depth = 0

    @contextmanager
    def locked(self):
        import fcntl
        with self._thread_lock:
            if self._depth == 0:
                self._lock_stream = (self.root / "mutation.lock").open("a")
                fcntl.flock(self._lock_stream, fcntl.LOCK_EX)
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0:
                    fcntl.flock(self._lock_stream, fcntl.LOCK_UN)
                    self._lock_stream.close()

    @serialized
    def enqueue(self, actor, *, step, suites, reason="periodic", priority=50, baseline=None):
        job_id = f"{int(priority):03d}_{int(step):010d}_{reason}_{uuid.uuid4().hex[:10]}"
        job = dict(job_id=job_id, actor=str(Path(actor).resolve()), vector_steps=int(step),
                   suites=list(suites), reason=reason, priority=priority, created_at=time.time(), baseline=baseline)
        write_json(self.root / "pending" / f"{job_id}.json", job)
        return job

    @serialized
    def claim(self, worker):
        for path in sorted((self.root / "pending").glob("*.json")):
            claimed = self.root / "running" / path.name
            try:
                # rename is atomic; a second claimant observes a missing source.
                os.rename(path, claimed)
            except FileNotFoundError:
                continue
            job = json.loads(claimed.read_text())
            job.update(worker=worker, worker_pid=os.getpid(), claimed_at=time.time())
            write_json(claimed, job)
            return job
        return None

    @serialized
    def finish(self, job, result):
        result = dict(result, job_id=job["job_id"], actor=job["actor"], vector_steps=job["vector_steps"],
                      reason=job["reason"], finished_at=time.time())
        write_json(self.root / "results" / f"{job['job_id']}.json", result)
        write_json(self.root / "summaries" / f"{job['job_id']}.json", evaluation_summary(result))
        write_json(self.root / "completed" / f"{job['job_id']}.json", dict(job, result_file=str(self.root / "results" / f"{job['job_id']}.json")))
        (self.root / "running" / f"{job['job_id']}.json").unlink(missing_ok=True)
        return result

    @serialized
    def fail(self, job, error):
        write_json(self.root / "failed" / f"{job['job_id']}.json", dict(job, error=str(error), failed_at=time.time()))
        (self.root / "running" / f"{job['job_id']}.json").unlink(missing_ok=True)
        # Failed evaluation protects the actor for a subsequent retry/review.
        self.protect(job["actor"], "evaluation_fault")

    @serialized
    def protect(self, actor, label):
        write_json(self.root / "protection" / f"{Path(actor).stem}__{label}.json",
                   dict(actor=str(Path(actor).resolve()), label=label, protected_at=time.time()))

    @serialized
    def protected(self):
        return {json.loads(p.read_text())["actor"] for p in (self.root / "protection").glob("*.json")}

    @serialized
    def assign(self, actor, label):
        """Move a best/latest label; immutable key-fault protections use protect()."""
        write_json(self.root / "protection" / f"label_{label}.json",
                   dict(actor=str(Path(actor).resolve()), label=label, protected_at=time.time()))

    @serialized
    def pending_actors(self):
        return {json.loads(p.read_text())["actor"] for name in ("pending", "running", "failed")
                for p in (self.root / name).glob("*.json")}

    @serialized
    def confirm_backup(self, actor, receipt):
        """The backup coordinator supplies a verified Drive receipt, never a guess."""
        actor = str(Path(actor).resolve())
        if receipt.get("verified") is not True or not any(receipt.get(key) for key in ("drive_file_id", "drive_url", "backup_receipt_path")):
            raise ValueError("Backup acknowledgement requires verified success and a concrete receipt reference")
        if not receipt.get("source_path") or str(Path(receipt["source_path"]).resolve()) != actor:
            raise ValueError("Backup receipt source_path does not identify this actor")
        info = Path(actor).stat()
        if receipt.get("source_size_bytes") != info.st_size or receipt.get("source_mtime_ns") != info.st_mtime_ns:
            raise ValueError("Backup receipt size/mtime identity does not identify the current immutable actor file")
        matching_jobs = [json.loads(path.read_text()) for path in (self.root / "completed").glob("*.json")]
        steps = {int(job["vector_steps"]) for job in matching_jobs if job["actor"] == actor}
        if len(steps) != 1 or receipt.get("vector_steps") != next(iter(steps)):
            raise ValueError("Backup receipt vector_steps does not match this actor's evaluation identity")
        changed = []
        for path in (self.root / "results").glob("*.json"):
            result = json.loads(path.read_text())
            if result.get("actor") == actor and result.get("important_backup_pending"):
                result.update(important_backup_pending=False, backup_receipt=receipt)
                write_json(path, result)
                write_json(self.root / "summaries" / path.name, evaluation_summary(result))
                changed.append(result["job_id"])
        return changed

    @serialized
    def prune_evaluated(self):
        """Never remove results/logs, protected weights, or unassessed actors."""
        protected = self.protected() | self.pending_actors()
        deleted, assessments = [], {}
        for path in (self.root / "completed").glob("*.json"):
            job = json.loads(path.read_text())
            actor = job["actor"]
            result = json.loads(Path(job["result_file"]).read_text())
            if not result.get("retention_evaluated", False) or result.get("important_backup_pending", False):
                protected.add(actor)
            assessments.setdefault(actor, []).append(job)
        for actor, jobs in assessments.items():
            if actor in protected or not Path(actor).exists():
                continue
            Path(actor).unlink()
            deleted.append(dict(actor=actor, job_ids=[job["job_id"] for job in jobs], deleted_at=time.time(),
                                reason="evaluated_non_best_unprotected; result_and_logs_preserved"))
        if deleted:
            with (self.root / "deletions.jsonl").open("a") as stream:
                stream.writelines(json.dumps(item) + "\n" for item in deleted)
        return deleted

