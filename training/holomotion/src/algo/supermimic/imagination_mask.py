# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Natural reference endings without invented motion transitions."""

import torch


def reference_masks(motion, start_frames, horizon):
    """Return valid transitions, natural terminals and valid critic states.

    A start at frame f of an N-frame clip has N-1-f real transitions. The
    last one earns its reward; subsequent tensor slots carry no learning.
    """
    remaining = motion.clip_end[motion.clip_id[start_frames]] - start_frames
    if (remaining < 1).any():
        raise ValueError("Imagination starts require a real next reference frame")
    time = torch.arange(horizon, device=start_frames.device)[:, None]
    valid = time < remaining[None]
    natural_terminal = time + 1 >= remaining[None]
    state_time = torch.arange(horizon + 1, device=start_frames.device)[:, None]
    state_valid = state_time < remaining[None]
    return valid, natural_terminal, state_valid
