# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Reuse a first SHA256 for unchanged large shards during bounded N profiles."""

import json
from pathlib import Path

from holomotion.src.utils.artifacts import fingerprint


def cached_fingerprint(path, cache_path):
    path, cache_path = Path(path).resolve(), Path(cache_path)
    info = path.stat()
    identity = dict(size=info.st_size, mtime_ns=info.st_mtime_ns, inode=info.st_ino)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    old = cache.get(str(path))
    if old and old.get("identity") == identity:
        return old["sha256"]
    digest = fingerprint(path)
    cache[str(path)] = dict(identity=identity, sha256=digest)
    from .evaluation_queue import write_json
    write_json(cache_path, cache)
    return digest

