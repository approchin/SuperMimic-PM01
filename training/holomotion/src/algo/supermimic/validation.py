# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Declare a separate immutable validation split without changing train inputs."""

import json
from pathlib import Path

from holomotion.src.utils.artifacts import fingerprint


def validation_contract(training_manifest, validation_manifest):
    training_manifest = Path(training_manifest).resolve()
    validation_manifest = Path(validation_manifest).resolve()
    if training_manifest == validation_manifest:
        raise ValueError("Training and validation manifests must differ")
    training = json.loads(training_manifest.read_text())
    validation = json.loads(validation_manifest.read_text())
    for key in ("dof_names", "root_quaternion_order", "motion_fps"):
        if training.get(key) != validation.get(key):
            raise ValueError(f"Validation reference contract differs: {key}")
    overlap = set(training["clips"]) & set(validation["clips"])
    if overlap:
        raise ValueError(f"Validation clip keys overlap training: {sorted(overlap)[:5]}")
    if not validation["clips"]:
        raise ValueError("Validation split is empty")
    paths = [validation_manifest]
    paths.extend((validation_manifest.parent / shard["file"]).resolve()
                 for shard in validation["hdf5_shards"])
    return dict(
        dataset_role="validation", manifest=str(validation_manifest),
        clips=len(validation["clips"]), clip_keys=sorted(validation["clips"]),
        input_sha256={str(path): fingerprint(path) for path in paths},
        split_policy="Existing separate manifests; no resplitting or filtering",
    )
