# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Lossless scalar event logging with one batched device transfer per flush."""

import json
import math
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class ScalarLogger:
    """Detach each scalar now; batch CUDA->host conversion and async disk writes.

    No events are sampled or dropped. Bounded writer backlog applies backpressure.
    Calls belong outside torch.compile/CUDA graph capture. Gradient finite checks
    in the learner remain immediate; flushing is not the numerical safety gate.
    """

    def __init__(self, path, *, flush_records=256, max_backlog=32):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_records = int(flush_records)
        self.context = {}
        self.pending = []
        self.ready = []
        self.events = 0
        self.module_counts = defaultdict(int)
        self.error = None
        self.closed = False
        self._queue = queue.Queue(maxsize=max_backlog)
        self._thread = threading.Thread(target=self._writer, name="011-scalar-writer", daemon=True)
        self._thread.start()

    def set_context(self, **values):
        self.context.update(values)

    def _check(self):
        if self.error is not None:
            raise RuntimeError("Scalar log writer failed; no further updates are safe") from self.error

    def _put(self, payload):
        while True:
            self._check()
            try:
                self._queue.put(payload, timeout=0.25)
                return
            except queue.Full:
                pass

    def emit(self, module, metrics, *, update_index=None, batch_index=None):
        self._check()
        if self.closed:
            raise RuntimeError("Scalar logger is already closed")
        event = dict(self.context, module=str(module), event_index=self.events, wall_time_unix=time.time())
        self.events += 1
        self.module_counts[str(module)] += 1
        if update_index is not None:
            event["update_index"] = int(update_index)
        if batch_index is not None:
            event["batch_index"] = int(batch_index)
        event["metrics"] = {}
        for key, value in metrics.items():
            if hasattr(value, "detach"):
                if value.numel() != 1:
                    raise ValueError(f"Log scalar {key!r} has shape {value.shape}")
                value = value.detach().reshape(()).clone()
            elif value is not None and not isinstance(value, (int, float, bool, str)):
                raise TypeError(f"Unsupported scalar {key}: {type(value)}")
            event["metrics"][key] = value
        self.pending.append(event)
        # The runner flushes at event boundaries; learner callbacks never
        # synchronize individual minibatches, including all twenty Critic fits.

    def flush(self, *, force=False):
        self._check()
        if not self.pending or (not force and len(self.pending) < self.flush_records):
            return []
        events, self.pending = self.pending, []
        groups = {}
        for event in events:
            for key, value in event["metrics"].items():
                if hasattr(value, "detach"):
                    groups.setdefault(str(value.device), []).append((event, key, value))
        if groups:
            import torch
            for items in groups.values():
                # One device synchronization/host transfer for the whole chunk.
                values = torch.stack([item[2].to(dtype=torch.float64) for item in items]).cpu().tolist()
                for (event, key, _), value in zip(items, values):
                    event["metrics"][key] = value
        self._put(events)
        self.ready.extend(events)
        return events

    def take_events(self):
        events, self.ready = self.ready, []
        return events

    def _writer(self):
        try:
            with self.path.open("a", buffering=1024 * 1024) as stream:
                while True:
                    chunk = self._queue.get()
                    try:
                        if chunk is None:
                            stream.flush()
                            return
                        stream.writelines(json.dumps(json_safe(event), allow_nan=False) + "\n" for event in chunk)
                        stream.flush()
                    finally:
                        self._queue.task_done()
        except BaseException as error:
            self.error = error

    def close(self):
        if self.closed:
            return
        self.flush(force=True)
        self.closed = True
        self._put(None)
        self._thread.join(timeout=30)
        self._check()
        if self._thread.is_alive():
            raise RuntimeError("Scalar writer did not finish flushing")

