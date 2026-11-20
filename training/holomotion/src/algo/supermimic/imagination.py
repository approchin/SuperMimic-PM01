# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
# Adapted from zita-ch/supermimic_vibed, a1508d42c9d1699a52e8233ab16d08c488441ab9.
# Copyright (c) 2026 ETH Zurich, Chong Zhang. MIT; see LICENSE.vibed.

"""Logical-batch imagination with bounded actor activation memory."""

import torch
from torch import nn

from holomotion.src.env.reference_tracking.mdp import (
    normalize_xi,
    unflatten_state,
)


def set_requires_grad(module, flag):
    for parameter in module.parameters():
        parameter.requires_grad_(flag)


def policy_update_microbatched(self):
    batch = self.args.pi_batch
    width = getattr(self.args, "pi_microbatch", batch)
    start, history = self.sample_valid_starts(batch)
    # Draw one logical population in the original time-major RNG order.
    # Execution chunks then reuse it without changing sampled noise.
    with torch.no_grad():
        prepared_history = self.mdp.add_noise(
            self.mdp.obs(
                unflatten_state(history["s"]),
                history["a_prev"],
                history["xi"],
                history["f"],
            )
        )
        prepared_noise = [
            self.mdp.add_noise(
                torch.zeros(batch, self.mdp.obs_dim, device=self.dev)
            )
            for _ in range(self.HZ)
        ]
    self.opt_pi.zero_grad(set_to_none=True)
    set_requires_grad(self.wm, False)
    set_requires_grad(self.critic, False)
    inputs, reward_batches, done_batches = [], [], []
    valid_batches, state_valid_batches = [], []
    finite_reference = self.mdp.robot_spec.kind == "pm01"
    total_valid = 0
    total_loss, total_reward, total_survival = 0.0, 0.0, 0.0
    logger = getattr(self, "scalar_logger", None)
    j_reward_sum = torch.zeros((), device=self.dev)
    j_tail_sum = torch.zeros_like(j_reward_sum)
    try:
        for first in range(0, batch, width):
            last = min(first + width, batch)
            size = last - first
            s = {key: value[first:last] for key, value in start.items()}
            state = unflatten_state(s["s"])
            initial_state = state
            context, frame, previous = s["xi"], s["f"], s["a_prev"]
            queue = s.get("delay_state")
            queues = [] if queue is None else [queue]
            normalized = normalize_xi(context)
            if finite_reference:
                from .imagination_mask import reference_masks

                valid, natural_terminal, state_valid = reference_masks(
                    self.motion, frame, self.HZ
                )
            else:
                valid = torch.ones(self.HZ, size, dtype=torch.bool, device=self.dev)
                natural_terminal = torch.zeros_like(valid)
                state_valid = torch.ones(
                    self.HZ + 1, size, dtype=torch.bool, device=self.dev
                )
            total_valid += int(valid.sum())
            goal = initial_goal = self.motion.goal(frame, self.K)
            observations = prepared_history[first:last]
            states, actions, previous_actions, frames, goals = (
                [],
                [],
                [],
                [],
                [],
            )
            for t in range(self.HZ):
                previous_actions.append(previous)
                result = self.imag_step(
                    state,
                    observations,
                    previous,
                    frame,
                    context,
                    normalized,
                    goal,
                    prepared_noise[t][first:last],
                    queue,
                )
                state, previous, observations, frame, goal = result[:5]
                if queue is not None:
                    queue = result[5]
                    queues.append(queue.detach())
                states.append(state)
                actions.append(previous)
                frames.append(frame)
                goals.append(goal)

            def flat(values):
                return torch.stack(values).flatten(0, 1)

            all_states = tuple(
                flat([value[i] for value in states]) for i in range(6)
            )
            reward, done = self.reward_term(
                all_states,
                flat(frames),
                flat(actions),
                flat(previous_actions),
            )
            reward, done = (
                reward.reshape(self.HZ, size),
                done.reshape(self.HZ, size),
            )
            # The final real transition retains its reward. Natural endings
            # stop return bootstrapping even when physical survival is high.
            reward = torch.where(valid, reward, torch.zeros_like(reward))
            done = torch.where(natural_terminal, torch.ones_like(done), done)
            alive = torch.cumprod(1 - done, 0)
            survival = torch.cat(
                (torch.ones(1, size, device=self.dev), alive[:-1]), 0
            )
            discount = (
                self.args.gamma
                ** torch.arange(self.HZ, device=self.dev, dtype=reward.dtype)[
                    :, None
                ]
            )
            objective = (discount * survival * reward).sum(0)
            j_reward_sum += objective.detach().sum() / batch
            if not self.args.no_critic:
                tail = self.args.gamma**self.HZ * alive[
                    -1
                ] * self.critic(
                    self._critic_input(state, frame, normalized, goal, queue)
                )
                j_tail_sum += tail.detach().sum() / batch
                objective = objective + tail
            loss = -objective.mean() * (size / batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite imagined objective")
            loss.backward()
            total_loss += float(loss.detach())
            total_reward += float((survival * reward).detach().sum()) / batch
            total_survival += float(alive[-1].detach().sum()) / batch
            if not self.args.no_critic:
                with torch.no_grad():
                    critic_states = tuple(
                        torch.cat(
                            (
                                initial_state[i][None],
                                torch.stack(
                                    [value[i].detach() for value in states]
                                ),
                            ),
                            0,
                        ).flatten(0, 1)
                        for i in range(6)
                    )
                    critic_frames = torch.cat(
                        (s["f"][None], torch.stack(frames)), 0
                    ).flatten()
                    critic_goal = torch.cat(
                        (initial_goal[None], torch.stack(goals)), 0
                    ).flatten(0, 1)
                    critic_context = (
                        normalized[None]
                        .expand(self.HZ + 1, -1, -1)
                        .flatten(0, 1)
                    )
                    inputs.append(
                        self._critic_input(
                            critic_states,
                            critic_frames,
                            critic_context,
                            critic_goal,
                            torch.stack(queues).flatten(0, 1) if queues else None,
                        ).reshape(self.HZ + 1, size, -1)
                    )
                    reward_batches.append(reward.detach())
                    done_batches.append(done.detach())
                    valid_batches.append(valid)
                    state_valid_batches.append(state_valid)
        gradient = nn.utils.clip_grad_norm_(
            self.pi.parameters(), 1.0, error_if_nonfinite=True
        )
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        self.opt_pi.step()
        if logger is not None:
            logger.emit("actor", {"loss":-(j_reward_sum+j_tail_sum), "Jreward":j_reward_sum, "Jtail":j_tail_sum,
                                  "grad_norm_preclip":gradient.detach(), "valid_imagination_steps":total_valid})
    finally:
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)
    value_loss = torch.zeros((), device=self.dev)
    if not self.args.no_critic:
        x = torch.cat(inputs, 1)
        rewards, dones = (
            torch.cat(reward_batches, 1),
            torch.cat(done_batches, 1),
        )
        valid = torch.cat(valid_batches, 1)
        state_valid = torch.cat(state_valid_batches, 1)
        self.critic.norm.update(x[state_valid])
        with torch.no_grad():
            values = self.critic(x)
            carry, targets = values[-1], []
            for t in range(self.HZ - 1, -1, -1):
                carry = rewards[t] + self.args.gamma * (1 - dones[t]) * (
                    (1 - self.args.lam) * values[t + 1] + self.args.lam * carry
                )
                targets.append(carry)
            target = torch.stack(targets[::-1])[valid]
        x = x[:-1][valid]
        critic_losses, critic_gradients = [], []
        for epoch in range(self.args.critic_epochs):
            for batch_index, ids in enumerate(torch.randperm(len(target), device=self.dev).chunk(
                min(self.args.critic_minibatches, len(target))
            )):
                pred = self.critic(x[ids])
                truth = target[ids]
                td = pred - truth
                value_loss = td.square().mean()
                self.opt_v.zero_grad(set_to_none=True)
                value_loss.backward()
                critic_gradient = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 1.0, error_if_nonfinite=True
                )
                self.opt_v.step()
                if logger is not None:
                    critic_losses.append(value_loss.detach())
                    critic_gradients.append(critic_gradient.detach())
                    with torch.no_grad():
                        logger.emit("critic", {"loss":value_loss.detach(), "grad_norm_preclip":critic_gradient.detach(),
                            "value_mean":pred.mean(), "value_max":pred.max(), "value_min":pred.min(),
                            "target_mean":truth.mean(), "target_max":truth.max(), "target_min":truth.min(),
                            "value_std":pred.std(unbiased=False), "target_std":truth.std(unbiased=False), "td_mse":value_loss.detach(),
                            "td_mean":td.mean(), "td_abs_mean":td.abs().mean(), "td_abs_max":td.abs().max(), "epoch":epoch},
                            batch_index=epoch*self.args.critic_minibatches+batch_index)
        if logger is not None and critic_losses:
            losses = torch.stack(critic_losses)
            grads = torch.stack(critic_gradients)
            logger.emit("critic_summary", {"loss_mean":losses.mean(), "loss_max":losses.max(),
                "gradient_norm_mean":grads.mean(),"gradient_norm_max":grads.max(),"minibatches":len(critic_losses)})

    self.last_valid_imagined_steps = total_valid

    def scalar(value):
        return torch.tensor(value, device=self.dev)

    return (
        scalar(total_loss),
        value_loss.detach(),
        scalar(total_reward / self.HZ / 0.1),
        gradient.detach(),
        scalar(total_survival),
    )
