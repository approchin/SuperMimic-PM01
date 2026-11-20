# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""The native 20 ms control period contains four 5 ms MuJoCo substeps."""
import torch

SUBSTEPS = 4


def reset_state(held, delay_ticks):
    return torch.cat((held, held, delay_ticks.to(held.dtype).unsqueeze(-1)), -1)


def desired_target(action, calibration, default_q, scale):
    return default_q + calibration + scale * action


def substep_target(previous_issued, new_target, delay_ticks, substep):
    """The previous issuance is due at the latest when this new period begins."""
    return torch.where(delay_ticks > substep, previous_issued, new_target)


def write_substep_target(previous_issued, new_target, delay_ticks, substep, *, mask, out):
    """Write native physics controls using storage kept alive across Warp graph replays."""
    torch.gt(delay_ticks, substep, out=mask)
    torch.where(mask, previous_issued, new_target, out=out)


def next_state(state, new_target):
    """End-of-interval held target, latest issuance and episode delay; all remain differentiable."""
    width = new_target.shape[-1]
    previous_issued = state[..., width:2 * width]
    ticks = state[..., -1:]
    held = substep_target(previous_issued, new_target, ticks, SUBSTEPS-1)
    return torch.cat((held, new_target, ticks), -1)
