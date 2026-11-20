# PM01 release note (2026-11-17): adapted or added for the PM01 integration.
"""Batched MuJoCo Warp physics for the reference-method experiment.

Each world has its own articulation, infinite floor, friction and torso COM.
Physical next states and the command queue are recorded before any reset.
"""

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp

from .mdp import MAX_STEP_REWARD, flatten_state
from .robot import G1_JOINT_NAMES, action_scale, default_joint_pos_list
from .rotations import mat_to_quat, quat_to_mat, so3_exp
from . import command_delay


class WarpTrackingBackend:
    def __init__(
        self,
        model,
        motion,
        mdp,
        num_envs,
        *,
        device="cuda:0",
        gamma=0.99,
        evaluation=False,
        max_episode_steps=500,
        nconmax=128,
        njmax=768,
        capture_graph=True,
        eval_delay_ms=0,
        stronger_reset_fraction=0.0,
    ):
        if num_envs < 1 or max_episode_steps < 1 or not 0 <= gamma < 1:
            raise ValueError("Invalid physical rollout dimensions")
        self.cpu_model, self.motion, self.mdp = model, motion, mdp
        self.robot_spec=mdp.robot_spec
        spec=self.robot_spec;n=spec.joints
        self.N, self.device, self.gamma = num_envs, device, gamma
        self.eval_mode = evaluation
        self.max_episode_steps = 10**9 if evaluation else max_episode_steps
        self.reset_noise = not evaluation
        self.eval_delay_ms = eval_delay_ms
        if eval_delay_ms not in (0, 5, 10, 15, 20) or stronger_reset_fraction not in (0., .1, .2):
            raise ValueError("Unsupported delay or stronger-reset fraction")
        self.stronger_reset_fraction = stronger_reset_fraction
        self.last_stronger_reset = torch.zeros(num_envs, device=device, dtype=torch.bool)
        wp.init()
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(device))
        self.default_q = torch.tensor(spec.default_q, device=device)
        self.alpha = torch.tensor(
            spec.action_scales, device=device
        )
        self._torso_id = model.body(spec.torso_body).id
        self._ground_id = model.geom("floor").id
        randomized_torso = spec.kind == "pm01" and not evaluation
        batch_sizes = {"geom_friction": num_envs, "body_ipos": num_envs}
        if randomized_torso:
            batch_sizes.update({k:num_envs for k in ("body_invweight0","dof_invweight0","actuator_acc0")})
        with wp.ScopedDevice(device), wp.ScopedStream(self.stream):
            self.model = mjw.put_model(
                model, batch_sizes=batch_sizes,
            )
            if randomized_torso:
                self.model.stat.meaninertia = wp.full(num_envs, float(model.stat.meaninertia), dtype=wp.float32, device=device)
            self.model.opt.warn_overflow = 0
            initial = mujoco.MjData(model)
            initial.qpos[:3] = [0, 0, 1.0]
            initial.qpos[7:] = self.default_q.cpu().numpy()
            mujoco.mj_forward(model, initial)
            self.data = mjw.put_data(
                model, initial, nworld=num_envs, nconmax=nconmax, njmax=njmax
            )
        self.qpos, self.qvel, self.ctrl = (
            wp.to_torch(value)
            for value in (self.data.qpos, self.data.qvel, self.data.ctrl)
        )
        self._friction = wp.to_torch(self.model.geom_friction)
        self._com = wp.to_torch(self.model.body_ipos)
        self._nominal_com = self._com[:, self._torso_id].clone()
        self._inertia_tables = {}
        if randomized_torso:
            from .com_constants import TorsoConstants
            self._torso_constants = TorsoConstants(model,self._torso_id,device)
            self._inertia_tables = {k:wp.to_torch(getattr(self.model,k)) for k in ("body_invweight0","dof_invweight0","actuator_acc0")}
            self._inertia_tables["meaninertia"] = wp.to_torch(self.model.stat.meaninertia)
        if spec.kind == "pm01":
            feet = {model.body(name).id for name in ("LINK_FOOT_L", "LINK_FOOT_R")}
            foot_geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] in feet and
                          (model.geom_contype[i] or model.geom_conaffinity[i])]
            if not foot_geoms or model.npair:
                raise ValueError("PM01 foot-floor contact must use native geom friction, without explicit pair overrides")
            self._foot_geoms = torch.tensor(foot_geoms, device=device)
            self.delay_state = torch.zeros(num_envs, spec.delay_state_dim, device=device)
            self._issued_target = torch.zeros_like(self.ctrl)
            self._delay_mask = torch.empty((num_envs, 1), dtype=torch.bool, device=device)
        else:
            self.delay_state = None
        self.xi = torch.zeros(num_envs, spec.context_dim, device=device)
        if not evaluation and spec.kind=='g1':
            self.xi[:, 1+n:] = torch.rand(
                num_envs, 3, device=device
            ) * torch.tensor([0.05, 0.1, 0.1], device=device) - torch.tensor(
                [0.025, 0.05, 0.05], device=device
            )
        self._com[:, self._torso_id] = self._nominal_com + self.xi[:, 1+n:]
        self.f = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ep_step = torch.zeros_like(self.f)
        self.a_prev = torch.zeros(num_envs, spec.applied_actions, device=device)
        self.start_bin = torch.zeros_like(self.f)
        self.ep_disc_ret = torch.zeros(num_envs, device=device)
        self.ep_ret = torch.zeros_like(self.ep_disc_ret)
        self.G_max = MAX_STEP_REWARD / (1 - gamma)
        self.bin_G = torch.zeros(motion.num_bins, device=device)
        self.bin_w_avg = torch.full_like(self.bin_G, 1 / motion.num_bins)
        self.source_sampler = None
        self.done_eval = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failed_eval = torch.zeros_like(self.done_eval)
        self._overflow_seen = torch.zeros(
            num_envs, dtype=torch.int32, device=device
        )
        self._native_overflow = wp.to_torch(self.data.overflow)
        # Fixed buffers also participate in graph replay, preserving the exact
        # native state before each substep if collision processing fails.
        self._substep_qpos = torch.empty(4, num_envs, model.nq, device=device)
        self._substep_qvel = torch.empty(4, num_envs, model.nv, device=device)
        self._substep_ctrl = torch.empty(4, num_envs, model.nu, device=device)
        self._substep_status = torch.zeros(
            4, num_envs, dtype=torch.int32, device=device
        )
        # MuJoCo Warp also stores ordinary bounded solver-iteration statuses in
        # its overflow bitset. These are distinct from truncated contact or
        # constraint buffers. Retain both; reject only capacity/invalid states.
        solver_limits = (
            mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
        )
        self.capacity_mask = int(mjw.OverflowType.ALL & ~solver_limits)
        self.graph = None
        self.reset(torch.arange(num_envs, device=device))
        # Eager compilation precedes graph capture. These preparation steps are
        # excluded from the physical training transition counter.
        with wp.ScopedDevice(device), wp.ScopedStream(self.stream):
            self._advance()
            if capture_graph:
                # Warp cannot begin capture on Torch's legacy default stream.
                # Capture on a registered Warp stream, then replay on the
                # caller's stream. Keep Torch bookkeeping in the same capture.
                torch.cuda.synchronize(device)
                self.capture_stream = wp.Stream(device)
                with (
                    torch.cuda.stream(wp.stream_to_torch(self.capture_stream)),
                    wp.ScopedStream(self.capture_stream),
                ):
                    with wp.ScopedCapture(
                        stream=self.capture_stream
                    ) as capture:
                        self._advance()
                self.graph = capture.graph
        self.reset(torch.arange(num_envs, device=device))
        self._overflow_seen.zero_()

    def _advance(self):
        for substep in range(4):
            if self.delay_state is not None:
                n = self.robot_spec.applied_actions
                command_delay.write_substep_target(
                    self.delay_state[:, n:2*n], self._issued_target,
                    self.delay_state[:, -1:], substep, mask=self._delay_mask, out=self.ctrl,
                )
            self._substep_qpos[substep].copy_(self.qpos)
            self._substep_qvel[substep].copy_(self.qvel)
            self._substep_ctrl[substep].copy_(self.ctrl)
            mjw.step(self.model, self.data)
            self._substep_status[substep].copy_(self._native_overflow)
            self._overflow_seen.bitwise_or_(self._native_overflow)

    def set_eval_delay(self, ids, delay_ms):
        if not self.eval_mode or self.delay_state is None:
            raise ValueError("Explicit delay override is for PM01 evaluation")
        ms = torch.as_tensor(delay_ms, device=self.device)
        if ((ms < 0) | (ms > 20) | (ms.remainder(5) != 0)).any():
            raise ValueError("Evaluation delay must be a supported 5 ms multiple")
        self.delay_state[ids, -1] = ms.to(self.delay_state.dtype) / 5.

    def state(self):
        p, quat = self.qpos[:, :3].clone(), self.qpos[:, 3:7].clone()
        rotation = quat_to_mat(quat)
        # MuJoCo's free-joint angular generalized velocity is body-local.
        angular = (rotation @ self.qvel[:, 3:6].unsqueeze(-1)).squeeze(-1)
        return (
            p,
            rotation,
            self.qvel[:, :3].clone(),
            angular,
            self.qpos[:, 7:].clone(),
            self.qvel[:, 6:].clone(),
        )

    def configure_sampling(self, index_path):
        from .source_sampling import SourceBinSampler
        self.source_sampler = SourceBinSampler(index_path, self.motion, self.device)
        self.source_sampler.initialize(self)
        self.reset(torch.arange(self.N, device=self.device))

    def sample_frames(self, n):
        motion = self.motion
        if self.source_sampler is not None:
            bins = self.source_sampler.sample(self, n)
            return self.frames_in_bins(bins), bins
        soft = torch.softmax(-25 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probabilities = (
            0.5 / motion.num_bins
            + 0.4 * soft
            + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        )
        bins = torch.multinomial(probabilities, n, replacement=True)
        return self.frames_in_bins(bins), bins

    def frames_in_bins(self, bins):
        motion = self.motion
        start = motion.bin_start[bins]
        end = torch.minimum(start + 50, motion.clip_end[motion.bin_clip[bins]])
        frame = (
            start
            + (
                torch.rand(len(bins), device=self.device) * (end - start).float()
            ).long()
        )
        return frame

    def reset(self, ids, frames=None):
        if not len(ids):
            return
        if frames is None:
            frames, bins = self.sample_frames(len(ids))
        else:
            bins = self.motion.frame_bin[frames]
        p, rotation, v, w, q, qd = (
            value.clone() for value in self.motion.ref_state(frames)
        )
        if self.reset_noise:

            def uniform(half):
                return (
                    2 * torch.rand(len(ids), len(half), device=self.device) - 1
                ) * torch.tensor(half, device=self.device)

            p += uniform([0.05, 0.05, 0.01])
            rotation = so3_exp(uniform([0.1, 0.1, 0.2])) @ rotation
            velocity_offset = uniform([0.5, 0.5, 0.2])
            self.last_stronger_reset[ids] = False
            if self.stronger_reset_fraction:
                stronger = torch.rand(len(ids), device=self.device) < self.stronger_reset_fraction
                angle = torch.rand(len(ids), device=self.device) * (2*torch.pi)
                speed = torch.rand(len(ids), 1, device=self.device) * .5 + .5
                planar = speed * torch.stack((torch.cos(angle), torch.sin(angle)), -1)
                velocity_offset[:, :2] = torch.where(stronger[:, None], planar, velocity_offset[:, :2])
                self.last_stronger_reset[ids] = stronger
            v += velocity_offset
            w += uniform([0.52, 0.52, 0.78])
            if self.robot_spec.kind == "pm01":
                # The exogenous head retains its original reference angle.
                q[:, :-1] = (q[:, :-1] + uniform([.1] * self.robot_spec.actions)).clamp(
                    self.mdp.kin.q_lo[:-1], self.mdp.kin.q_hi[:-1])
            else:
                q += uniform([0.1] * self.robot_spec.joints)
        if self.robot_spec.kind=='g1':q = q.clamp(self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        mask = torch.zeros(self.N, dtype=torch.bool, device=self.device)
        mask[ids] = True
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            mjw.reset_data(self.model, self.data, wp.from_torch(mask))
        self.qpos[ids] = torch.cat((p, mat_to_quat(rotation), q), -1)
        self.qvel[ids] = torch.cat(
            (
                v,
                (rotation.transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1),
                qd,
            ),
            -1,
        )
        self.ctrl[ids] = q
        if self.delay_state is not None:
            ticks = (torch.full((len(ids),), self.eval_delay_ms//5, device=self.device) if self.eval_mode else
                     torch.randint(0, 5, (len(ids),), device=self.device))
            self.delay_state[ids] = command_delay.reset_state(q, ticks)
            self._issued_target[ids] = q
        self.f[ids], self.ep_step[ids], self.a_prev[ids] = frames, 0, 0.0
        self.start_bin[ids] = bins
        if self.source_sampler is not None:
            self.source_sampler.count(self.source_sampler.reset_counts,self.motion.clip_id[frames])
        self.ep_disc_ret[ids], self.ep_ret[ids] = 0.0, 0.0
        n=self.robot_spec.joints
        if self.eval_mode:
            self.xi[ids, :1+n] = 0.0
            self.xi[ids, 0] = 1.0
        else:
            self.xi[ids, 0] = torch.empty(
                len(ids), device=self.device
            ).uniform_(0.3, 1.2)
            self.xi[ids, 1:1+n] = torch.empty(
                len(ids), n, device=self.device
            ).uniform_(-0.01, 0.01)
        if self.robot_spec.kind == "pm01":
            if self.eval_mode:
                self.xi[ids, 1+n:] = 0.
            else:
                radius = self._nominal_com.new_tensor([.025, .05, .05])
                self.xi[ids, 1+n:] = (2*torch.rand(len(ids), 3, device=self.device)-1)*radius
                constants = self._torso_constants.compute(self.xi[ids, 1+n:])
                for key, values in constants.items():
                    self._inertia_tables[key][ids] = values
            self._com[ids, self._torso_id] = self._nominal_com[ids] + self.xi[ids, 1+n:]
            # In equal-priority native contacts friction uses the larger coefficient. Set both partners.
            # Historical clean evaluation assigned 1 to every geom, including robot-to-robot contacts.
            if self.eval_mode:
                self._friction[ids, :, 0] = 1.
            else:
                self._friction[ids[:, None], self._foot_geoms[None, :], 0] = self.xi[ids, :1]
                self._friction[ids, self._ground_id, 0] = self.xi[ids, 0]
        else:
            self._friction[ids, :, 0] = self.xi[ids, :1]
            self._friction[ids, self._ground_id, 0] = 1. if self.eval_mode else .3

    def _failure_diagnostic(self, state, frame, previous, context, step, policy_action, action):
        bad = (self._overflow_seen & self.capacity_mask) != 0
        for value in (self.qpos, self.qvel, action):
            bad |= ~torch.isfinite(value).all(dim=-1)
        ids = bad.nonzero().flatten()
        clip = self.motion.clip_id[frame]
        selected_clips = clip[ids].tolist()
        return dict(
            format="pm01_physics_failure_v2",
            affected_worlds=ids.clone(),
            status_bits=self._overflow_seen.clone(),
            status_meaning={name: int(value) for name, value in mjw.OverflowType.__members__.items()},
            qpos=self.qpos.clone(),
            qvel=self.qvel.clone(),
            substep_qpos_before=self._substep_qpos.clone(),
            substep_qvel_before=self._substep_qvel.clone(),
            substep_pd_target=self._substep_ctrl.clone(),
            substep_status=self._substep_status.clone(),
            pre_state=flatten_state(state),
            action=action.clone(),
            policy_action=policy_action.clone(),
            previous_applied_action=previous,
            pd_target=self.ctrl.clone(),
            context=context,
            reference_frame=frame,
            local_reference_frame=frame - self.motion.clip_start[clip],
            reference_clip_id=clip.clone(),
            affected_clip_names=[self.motion.names[i] for i in selected_clips],
            episode_step=step,
            head_reference=self.motion.q[frame, -1:].clone(),
            physics_dt_s=float(self.cpu_model.opt.timestep),
            control_substeps=4,
            active_constraints=wp.to_torch(self.data.nefc).clone(),
            contacts=wp.to_torch(self.data.nacon).clone(),
            failed_transition_entered_replay=False,
        )

    @torch.no_grad()
    def step(self, action):
        policy_action=action.clone()
        head_reference=self.motion.q[self.f,-1:].clone()
        action=self.robot_spec.compose_action(action,self.motion.q[self.f],self.xi)
        if self.eval_mode:
            # Worlds are independent. Completed trials contribute no further
            # metrics and stay at a neutral, finite preparation state.
            ids = self.done_eval.nonzero().flatten()
            self.qpos[ids, :3] = torch.tensor(
                [0.0, 0.0, 1.0], device=self.device
            )
            self.qpos[ids, 3:7] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device
            )
            self.qpos[ids, 7:] = self.default_q
            self.qvel[ids] = 0.0
            action = torch.where(self.done_eval[:, None], 0.0, action)
        state, frame = self.state(), self.f.clone()
        previous, context, step = (
            self.a_prev.clone(),
            self.xi.clone(),
            self.ep_step.clone(),
        )
        target = command_delay.desired_target(action, context[:, 1:1+self.robot_spec.joints], self.default_q, self.alpha)
        old_queue = None
        if self.delay_state is None:
            self.ctrl.copy_(target)
        else:
            old_queue = self.delay_state.clone()
            self._issued_target.copy_(target)
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            if self.graph is None:
                self._advance()
            else:
                wp.capture_launch(self.graph, stream=self.stream)
        next_state = self.state()
        if self.source_sampler is not None:
            self.source_sampler.count(self.source_sampler.transition_counts,self.motion.clip_id[frame])
        if old_queue is not None:
            self.delay_state.copy_(command_delay.next_state(old_queue, target))
        status_failed = bool((self._overflow_seen & self.capacity_mask).any())
        state_failed = not all(torch.isfinite(value).all() for value in next_state)
        if status_failed or state_failed:
            packet = self._failure_diagnostic(
                state, frame, previous, context, step, policy_action, action
            )
            ids = packet["affected_worlds"]
            detail = dict(
                worlds=ids.tolist(),
                status_bits=self._overflow_seen[ids].tolist(),
                clips=packet["affected_clip_names"],
                local_reference_frames=packet["local_reference_frame"][ids].tolist(),
            )
            error_type = RuntimeError if status_failed else FloatingPointError
            error = error_type(f"MuJoCo Warp invalid physical transition: {detail}")
            error.physics_diagnostic = packet
            raise error
        # Finished evaluation worlds are kept at their last reference frame.
        next_frame = torch.minimum(
            frame + 1, self.motion.clip_end[self.motion.clip_id[frame]]
        )
        reward, terms = self.mdp.reward(
            next_state, next_frame, action, previous, terms=True
        )
        terminated = self.mdp.termination(next_state, next_frame,root_quaternion_wxyz=self.qpos[:,3:7])
        timeout = (
            next_frame >= self.motion.clip_end[self.motion.clip_id[frame]]
        ) | (step + 1 >= self.max_episode_steps)
        done = terminated | timeout
        out = dict(
            s=flatten_state(state),
            a=action.clone(),
            a_prev=previous,
            xi=context,
            f=frame,
            s1=flatten_state(next_state),
            r=reward,
            term=terminated,
            done=done,
            ep_step=step,
            terms=terms,
        )
        if self.robot_spec.kind=='pm01':
            out.update(a_policy=policy_action, head_reference=head_reference, pd_target=target,
                       last_applied_pd_target=self.ctrl.clone(),delay_state=old_queue, truncated=timeout.clone())
        self.ep_disc_ret += self.gamma ** step.float() * reward
        self.ep_ret += reward
        self.f, self.ep_step, self.a_prev = (
            next_frame,
            step + 1,
            action.clone(),
        )
        if self.eval_mode:
            self.failed_eval |= terminated & ~self.done_eval
            self.done_eval |= done
            return out
        ids = done.nonzero().flatten()
        if len(ids):
            length = self.ep_step[ids].float()
            bootstrap = torch.where(
                terminated[ids],
                0.0,
                self.gamma**length
                * self.ep_ret[ids]
                / length
                / (1 - self.gamma),
            )
            returns = self.ep_disc_ret[ids] + bootstrap
            bins = self.start_bin[ids]
            # Parallel episodes can update the same bin. Aggregate first so
            # duplicate scatter writes cannot silently discard other returns.
            if self.source_sampler is not None:
                self.source_sampler.update(self,bins,returns)
            else:
                totals, counts = torch.zeros_like(self.bin_G), torch.zeros_like(self.bin_G)
                totals.scatter_add_(0, bins, returns)
                counts.scatter_add_(0, bins, torch.ones_like(returns))
                used = counts > 0
                self.bin_G[used] = .8*self.bin_G[used]+.2*totals[used]/counts[used]
            out["ep_len"], out["ep_ret_norm"] = (
                length,
                self.ep_ret[ids] / (length * MAX_STEP_REWARD),
            )
            self.reset(ids)
        return out

    def status_summary(self, *, include_world_bits=True):
        result = dict(
            capacity_overflow_worlds_seen=int(
                ((self._overflow_seen & self.capacity_mask) != 0).sum()
            ),
            solver_limit_worlds_seen=int(
                ((self._overflow_seen & ~self.capacity_mask) != 0).sum()
            ),
        )
        if include_world_bits:
            result.update(
                physics_status_bits=self._overflow_seen.tolist(),
                capacity_overflow_bits=(
                    self._overflow_seen & self.capacity_mask
                ).tolist(),
            )
        return result
