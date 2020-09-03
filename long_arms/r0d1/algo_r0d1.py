# ============================================================================
# The optionally recurrent Deep Q learning agent
# ============================================================================
import torch
from collections import namedtuple

from rlpyt.algos.dqn.dqn import DQN, SamplesToBuffer
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import namedarraytuple
from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done, discount_return_n_step
from rlpyt.utils.buffer import buffer_to, buffer_method, torchify_buffer

# TODO: update this
OptInfo = namedtuple("OptInfo", ["loss", "gradNorm",
                                 "t1_q_min", "t1_q_max", "t2_q_min", "t2_q_max",
                                 "t3_q_min", "t3_q_max", "tneg2_q_min", "tneg2_q_max",
                                 "final_q_min", "final_q_max",
                                 "t2_abs_delta", "final_abs_delta",
                                 "t2_true_abs_delta", "t3_true_abs_delta",
                                 "tneg2_true_abs_delta", "final_true_abs_delta",
                                 "avg_true_abs_delta",
                                 "t2_grad_bptt_norm", "t2_grad_curt_norm",
                                 "tneg2_grad_bptt_norm", "tneg2_grad_curt_norm",
                                 "final_grad_bptt_norm", "final_grad_curt_norm",
                                 "avg_grad_bptt_norm", "avg_grad_curt_norm",
                                 "t2_grad_cossim", "tneg2_grad_cossim",
                                 "final_grad_cossim", "avg_grad_cossim",
                                 "tdAbsErr", "priority"])

SamplesToBufferRnn = namedarraytuple("SamplesToBufferRnn",
                                     SamplesToBuffer._fields + (
                                         "prev_rnn_state",
                                     ))
PrioritiesSamplesToBuffer = namedarraytuple("PrioritiesSamplesToBuffer",
                                            ["priorities", "samples"])


class R0D1(DQN):
    """
    DEPRECATED COMMENT: Recurrent-replay DQN with options for: Double-DQN,
    Dueling Architecture, n-step returns, prioritized_replay.
    """

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            discount=0.997,
            lambda_coef=1.0,
            batch_T=12,  # replay trajectory length
            batch_B=64,
            warmup_T=0,  # originally 40
            store_rnn_state_interval=9,  # 0 for none, 1 for all. default was 40
            min_steps_learn=int(1e5),
            delta_clip=None,  # Typically use squared-error loss (Steven).
            replay_size=int(1e6),
            replay_ratio=1,
            target_update_interval=2500,  # (Steven says 2500 but maybe faster.)
            n_step_return=1,  # originally 5, minimum is 1
            learning_rate=1e-4,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,
            clip_grad_norm=80.,  # 80 (Steven)
            eps_steps=int(1e6),  # STILL IN ALGO; conver to itr, give to agent.
            double_dqn=False,  # originally True
            prioritized_replay=True,
            pri_alpha=0.6,
            pri_beta_init=0.9,
            pri_beta_final=0.9,
            pri_beta_steps=int(50e6),
            pri_eta=0.9,
            default_priority=None,
            input_priorities=False,  # default True, not sure what it is used for
            input_priority_shift=None,
            value_scale_eps=1e-3,  # 1e-3 (Steven).
            ReplayBufferCls=None,  # leave None to select by above options
            updates_per_sync=1,  # For async mode only.
    ):
        """
        :param discount:
        :param lambda_coef: lambda return coefficient
        :param delta_clip:
        :param target_update_interval:
        :param learning_rate:
        :param OptimCls:
        :param optim_kwargs:
        :param initial_optim_state_dict:
        :param clip_grad_norm:
        :param eps_steps:
        :param double_dqn:
        :param value_scale_eps:
        :param ReplayBufferCls:
        """
        if optim_kwargs is None:
            optim_kwargs = dict(eps=1e-3)  # Assumes Adam.
        if default_priority is None:
            default_priority = delta_clip or 1.
        # if input_priority_shift is None:  # only used in prioritized replay and warmup i think NOTE
        #     input_priority_shift = warmup_T // store_rnn_state_interval
        save__init__args(locals())
        self._batch_size = (self.batch_T + self.warmup_T) * self.batch_B

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """Similar to DQN but uses replay buffers which return sequences, and
        stores the agent's recurrent state."""
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )

        if self.store_rnn_state_interval > 0:
            example_to_buffer = SamplesToBufferRnn(
                *example_to_buffer,
                prev_rnn_state=examples["agent_info"].prev_rnn_state,
            )

        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
            rnn_state_interval=self.store_rnn_state_interval,
            # batch_T fixed for prioritized, (relax if rnn_state_interval=1 or 0).
            batch_T=self.batch_T + self.warmup_T,
        )

        ReplayCls = UniformSequenceReplayBuffer

        if self.ReplayBufferCls is not None:
            ReplayCls = self.ReplayBufferCls
            logger.log(f"WARNING: ignoring internal selection logic and using"
                       f" input replay buffer class: {ReplayCls} -- compatibility not"
                       " guaranteed.")
        self.replay_buffer = ReplayCls(**replay_kwargs)

        return self.replay_buffer

    def samples_to_buffer(self, samples):
        samples_to_buffer = super().samples_to_buffer(samples)
        if self.store_rnn_state_interval > 0:
            samples_to_buffer = SamplesToBufferRnn(
                *samples_to_buffer,
                prev_rnn_state=samples.agent.agent_info.prev_rnn_state,
            )
        if self.input_priorities:
            priorities = self.compute_input_priorities(samples)
            samples_to_buffer = PrioritiesSamplesToBuffer(
                priorities=priorities, samples=samples_to_buffer)
        return samples_to_buffer

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Similar to DQN, except allows to compute the priorities of new samples
        as they enter the replay buffer (input priorities), instead of only once they are
        used in training (important because the replay-ratio is quite low, about 1,
        so must avoid un-informative samples).
        """

        # ==
        # Store sample to buffer
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)

        # Initialize logging
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info

        # For evaluation only: compute the delta to true value prediction
        true_sample_delta = self.compute_true_delta(samples)
        true_abs_delta = abs(true_sample_delta)

        # ==
        # Train
        for _ in range(self.updates_per_optimize):
            # Get sample from buffer
            buffer_samples = self.replay_buffer.sample_batch(self.batch_B)

            # One training step
            self.optimizer.zero_grad()
            loss, info_dict = self.loss(buffer_samples)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.clip_grad_norm
            )
            self.optimizer.step()

            # Optionally compute and log gradient statistics
            log_gradients = True
            if log_gradients and self.agent.model.use_recurrence:
                # Extract the tensor of gradients (in the valid interval)
                pred_grad_list = [self.agent.model.prev_hs_pre_grad[i] for
                                  i in range(self.store_rnn_state_interval)]
                recu_grad_list = [self.agent.model.prev_hs_rec_grad[i] for
                                  i in range(self.store_rnn_state_interval)]
                pred_grads = torch.cat(pred_grad_list, dim=0)  # [T, B, d]
                recu_grads = torch.cat(recu_grad_list, dim=0)  # [T, B, d]

                # Compute the gradient norms of this minibatch
                curT_grad_norms = torch.norm(pred_grads, dim=2)  # [T, B]
                curT_grad_norms = torch.mean(curT_grad_norms, dim=1)  # [T]
                avg_curT_grad_norms = torch.mean(curT_grad_norms)
                bptT_grad_norms = torch.norm(recu_grads, dim=2)  # [T, B]
                bptT_grad_norms = torch.mean(bptT_grad_norms, dim=1)  # [T]
                avg_bptT_grad_norms = torch.mean(bptT_grad_norms)

                # Compute the cosine similarity
                cosnn = torch.nn.CosineSimilarity(dim=2, eps=1e-10)
                cossim = cosnn(pred_grads, recu_grads)  # [T, B]
                cossim = torch.mean(cossim, dim=1)  # [T]
                avg_cossim = torch.mean(cossim)

                # Logging the gradient interference statistics
                getattr(opt_info, "t2_grad_curt_norm").append(curT_grad_norms[1].item())
                getattr(opt_info, "tneg2_grad_curt_norm").append(curT_grad_norms[-2].item())
                getattr(opt_info, "final_grad_curt_norm").append(curT_grad_norms[-1].item())
                getattr(opt_info, "avg_grad_curt_norm").append(avg_curT_grad_norms.item())

                getattr(opt_info, "t2_grad_bptt_norm").append(bptT_grad_norms[1].item())
                getattr(opt_info, "tneg2_grad_bptt_norm").append(bptT_grad_norms[-2].item())
                getattr(opt_info, "final_grad_bptt_norm").append(bptT_grad_norms[-1].item())
                getattr(opt_info, "avg_grad_bptt_norm").append(avg_bptT_grad_norms.item())

                getattr(opt_info, "t2_grad_cossim").append(cossim[1].item())
                getattr(opt_info, "tneg2_grad_cossim").append(cossim[-2].item())
                getattr(opt_info, "final_grad_cossim").append(cossim[-1].item())
                getattr(opt_info, "avg_grad_cossim").append(avg_cossim.item())

            # Logging information
            opt_info.loss.append(loss.item())
            opt_info.gradNorm.append(grad_norm.clone().detach().item())

            for k in info_dict:
                if not hasattr(opt_info, k):
                    continue
                # Add to NamedTuple
                getattr(opt_info, k).append(info_dict[k].item())

            getattr(opt_info, "t2_true_abs_delta").append(true_abs_delta[1].item())
            getattr(opt_info, "t3_true_abs_delta").append(true_abs_delta[2].item())
            getattr(opt_info, "tneg2_true_abs_delta").append(true_abs_delta[-2].item())
            getattr(opt_info, "final_true_abs_delta").append(true_abs_delta[-1].item())
            getattr(opt_info, "avg_true_abs_delta").append(torch.mean(true_abs_delta).item())

            # Update counter
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target()

        # NOTE: only required for prioritized replay I think?
        # self.update_itr_hyperparams(itr)

        return opt_info

    def loss(self, samples):
        """
        Compute TD loss for online samples
        :param samples: SamplesFromReplay object
        :return:
        """

        # ==
        # Extract individual tensors from the sample, shape (T, B, *)

        # Input for LSTM
        # NOTE both networks take the previous action and rewards as
        #            LSTM input. May want to set these to zero.
        # TODO: set the reward and action to zero? or at least reawrd to zero
        #
        all_observation, all_action, all_reward = buffer_to(
            (samples.all_observation.clone().detach(),
             samples.all_action.clone().detach(),
             samples.all_reward.clone().detach()),
            device=self.agent.device)

        # all_action = torch.zeros(all_action.size())
        # all_reward = torch.zeros(all_reward.size())  # TODO make this a feature in future?

        # Extract action, reward and done on CPU
        action = samples.all_action[1:self.batch_T + 1]
        return_ = samples.return_[0:self.batch_T]  # "n-step" rewards, n=1
        done_n = samples.done_n[0:self.batch_T]

        # ==
        # Compute Q estimates (NOTE: no RNN warm-up)
        agent_slice = slice(0, self.batch_T)
        agent_inputs = AgentInputs(
            observation=all_observation[agent_slice].clone().detach(),
            prev_action=all_action[agent_slice].clone().detach(),
            prev_reward=all_reward[agent_slice].clone().detach(),
        )
        target_slice = slice(0, None)  # Same start t as agent. (0 + bT + nsr)
        target_inputs = AgentInputs(
            observation=all_observation[target_slice],
            prev_action=all_action[target_slice],
            prev_reward=all_reward[target_slice],
        )

        # Init RNN state should be 0s / None either way
        if self.store_rnn_state_interval == 0:
            init_rnn_state = None
        else:
            # [B,N,H]-->[N,B,H] cudnn.
            init_rnn_state = buffer_method(samples.init_rnn_state,
                                           "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state,
                                           "contiguous")
        target_rnn_state = init_rnn_state  # NOTE: no RNN warmup for target

        # Behavioural net Q estimate
        qs, _ = self.agent(*agent_inputs, init_rnn_state)  # [T,B,A]
        q = select_at_indexes(action, qs)
        # Target network Q estimates
        with torch.no_grad():
            target_qs, _ = self.agent.target(*target_inputs, target_rnn_state)
            if self.double_dqn:
                next_qs, _ = self.agent(*target_inputs, init_rnn_state)
                next_a = torch.argmax(next_qs, dim=-1)
                target_q = select_at_indexes(next_a, target_qs)
            else:
                target_q = torch.max(target_qs, dim=-1).values
            target_q = target_q[-self.batch_T:]  # Same length as q.

        # ==
        # Compute lambda return from valid sequence
        valid = valid_from_done(done_n)
        lambda_G = self.compute_lambda_return(return_, target_q,
                                              valid)  # (T, 1)

        # ==
        # Compute Losses
        delta = lambda_G - q
        losses = 0.5 * delta ** 2
        abs_delta = abs(delta)

        # NOTE: by default, with R2D1, use squared-error loss, delta_clip=None.
        if self.delta_clip is not None:  # Huber loss.
            b = self.delta_clip * (abs_delta - self.delta_clip / 2)
            losses = torch.where(abs_delta <= self.delta_clip, losses, b)

        loss = valid_mean(losses, valid)

        td_abs_errors = abs_delta.detach()
        # NOTE: not computing prioritization

        # ==
        # Compute information to log
        valid_t = int(torch.sum(valid[:, 0]).item())  # NOTE: assume all same length
        info_dict = {}
        # Store the q estimates (of the first sample in batch)
        qs_tensor = qs.clone().detach()  # [sample_T, sample_B, A]
        qs_tensor = qs_tensor[0:valid_t, :, :]  # [valid_t, sample_B, A]

        info_dict['t1_q_min'] = torch.min(qs_tensor[0, 0, :])
        info_dict['t1_q_max'] = torch.max(qs_tensor[0, 0, :])
        info_dict['t2_q_min'] = torch.min(qs_tensor[1, 0, :])
        info_dict['t2_q_max'] = torch.max(qs_tensor[1, 0, :])
        info_dict['t3_q_min'] = torch.min(qs_tensor[2, 0, :])
        info_dict['t3_q_max'] = torch.max(qs_tensor[2, 0, :])
        info_dict['tneg2_q_min'] = torch.min(qs_tensor[-2, 0, :])
        info_dict['tneg2_q_max'] = torch.max(qs_tensor[-2, 0, :])
        info_dict['final_q_min'] = torch.min(qs_tensor[-1, 0, :])
        info_dict['final_q_max'] = torch.max(qs_tensor[-1, 0, :])
        # Store the TD error (average over batch)
        delta_cp = delta.clone().detach()[0:valid_t]
        delta_tensor = torch.mean(delta_cp, dim=1)
        abs_delta_tensor = torch.mean(torch.abs(delta_cp), dim=1)
        info_dict['t2_abs_delta'] = abs_delta_tensor[1]
        info_dict['final_abs_delta'] = abs_delta_tensor[-1]
        # All abs errors
        # info_dict['valid_td_abs_errors'] = td_abs_errors * valid

        return loss, info_dict

    def compute_lambda_return(self, r_traj, v_traj, valid):
        """
        Compute the lambda return. Assumes valid gives the valid trajectory
        up to the very first "done"
        :param r_traj: reward traj, torch.tensor size (sample_T, 1)
        :param v_traj: value esti traj, torch.tensor size (sample_T, 1)
        :param valid: valid indicator, torch.tensor size (sample_T, 1)
        :return: G_traj: lambda return, torch.tensor size (T, 1)
        """

        # Initialize lambda tensor and coefficients
        lamb_G = torch.zeros(v_traj.size())
        lamb = self.lambda_coef
        gamma = self.discount

        # Compute lambda return via dynamic programming
        for t in reversed(range(lamb_G.size(0) - 1)):
            G_t = (r_traj[t, :] + (valid[t + 1, :] * (
                    ((1 - lamb) * gamma * v_traj[t + 1, :]) +
                    (lamb * gamma * lamb_G[t + 1, :]))
                                   ))
            lamb_G[t, :] = G_t * valid[t, :]

        return lamb_G

    def compute_true_delta(self, samples):
        """
        Helper method with no training purpose. Only purpose is to compute the
        "true" return as samples come in, make the current Q estimate and
        see what the difference is (i.e. for evaluation and logging only)

        NOTE: if multiple trajectories are collected in a single sample,
              only the first trajectory will be used.
        :param samples: samples from environment sampler
        :return: tensor of delta between true G and predicted Q
        """

        # Extract information to estimate Q
        all_observation, all_action, all_reward = buffer_to(
            (samples.env.observation.clone().detach(),
             samples.agent.prev_action.clone().detach(),
             samples.env.prev_reward.clone().detach()),
            device=self.agent.device)
        action = samples.agent.prev_action[1:self.batch_T + 1]
        done_n = samples.env.done[0:self.batch_T]

        agent_slice = slice(0, self.batch_T)
        agent_inputs = AgentInputs(
            observation=all_observation[agent_slice].clone().detach(),
            prev_action=all_action[agent_slice].clone().detach(),
            prev_reward=all_reward[agent_slice].clone().detach(),
        )

        # Note: no RNN initial states (NOTE: might be a bug later if I
        #       want to use partial trajectories)
        init_rnn_state = None

        # Estimate Q
        with torch.no_grad():
            qs, _ = self.agent(*agent_inputs, init_rnn_state)  # [T,B,A]
            q = select_at_indexes(action, qs)

        # Valid length
        valid = valid_from_done(done_n)
        valid_T = int(torch.sum(valid))

        # ==
        # Compute true return (highly specific to the delay action.py env)
        # NOTE: this is built specifically for the action independent, pure
        #       prediction variant of the delayed_actions.py env
        arm_num = int(samples.env.env_info.arm_num[(valid_T-1)])
        true_R = 1.0 if (arm_num == 1) else -1.0

        true_G = torch.zeros((valid_T, 1))
        true_G[-1] = true_R
        for i in reversed(range(valid_T-1)):
            true_G[i] = self.discount * true_G[i+1]
        true_G[0] = 0.0  # first state has expected 0

        # ==
        # Compute delta to true value
        true_delta = true_G - q[:valid_T]

        return true_delta


    # ==
    # Below is old code
    # ==

    def compute_input_priorities(self, samples):
        """Used when putting new samples into the replay buffer.  Computes
        n-step TD-errors using recorded Q-values from online network and
        value scaling.  Weights the max and the mean TD-error over each sequence
        to make a single priority value for that sequence.

        Note:
            Although the original R2D2 implementation used the entire
            80-step sequence to compute the input priorities, we ran R2D1 with 40
            time-step sample batches, and so computed the priority for each
            80-step training sequence based on one of the two 40-step halves.
            Algorithm argument ``input_priority_shift`` determines which 40-step
            half is used as the priority for the 80-step sequence.  (Since this
            method might get executed by alternating memory copiers in async mode,
            don't carry internal state here, do all computation with only the samples
            available in input.  Could probably reduce to one memory copier and keep
            state there, if needed.)
        """

        # """Just for first input into replay buffer.
        # Simple 1-step return TD-errors using recorded Q-values from online
        # network and value scaling, with the T dimension reduced away (same
        # priority applied to all samples in this batch; whereever the rnn state
        # is kept--hopefully the first step--this priority will apply there).
        # The samples duration T might be less than the training segment, so
        # this is an approximation of an approximation, but hopefully will
        # capture the right behavior.
        # UPDATE 20190826: Trying using n-step returns.  For now using samples
        # with full n-step return available...later could also use partial
        # returns for samples at end of batch.  35/40 ain't bad tho.
        # Might not carry/use internal state here, because might get executed
        # by alternating memory copiers in async mode; do all with only the
        # samples avialable from input."""
        samples = torchify_buffer(samples)
        q = samples.agent.agent_info.q
        action = samples.agent.action
        q_max = torch.max(q, dim=-1).values
        q_at_a = select_at_indexes(action, q)
        return_n, done_n = discount_return_n_step(
            reward=samples.env.reward,
            done=samples.env.done,
            n_step=self.n_step_return,
            discount=self.discount,
            do_truncated=False,  # Only samples with full n-step return.
        )
        # y = self.value_scale(
        #     samples.env.reward[:-1] +
        #     (self.discount * (1 - samples.env.done[:-1].float()) *  # probably done.float()
        #         self.inv_value_scale(q_max[1:]))
        # )
        nm1 = max(1, self.n_step_return - 1)  # At least 1 bc don't have next Q.
        y = self.value_scale(return_n +
                             (1 - done_n.float()) * self.inv_value_scale(q_max[nm1:]))
        delta = abs(q_at_a[:-nm1] - y)
        # NOTE: by default, with R2D1, use squared-error loss, delta_clip=None.
        if self.delta_clip is not None:  # Huber loss.
            delta = torch.clamp(delta, 0, self.delta_clip)
        valid = valid_from_done(samples.env.done[:-nm1])
        max_d = torch.max(delta * valid, dim=0).values
        mean_d = valid_mean(delta, valid, dim=0)  # Still high if less valid.
        priorities = self.pri_eta * max_d + (1 - self.pri_eta) * mean_d  # [B]
        return priorities.numpy()

    def value_scale(self, x):
        """Value scaling function to handle raw rewards across games (not clipped)."""
        return (torch.sign(x) * (torch.sqrt(abs(x) + 1) - 1) +
                self.value_scale_eps * x)

    def inv_value_scale(self, z):
        """Invert the value scaling."""
        return torch.sign(z) * (((torch.sqrt(1 + 4 * self.value_scale_eps *
                                             (abs(z) + 1 + self.value_scale_eps)) - 1) /
                                 (2 * self.value_scale_eps)) ** 2 - 1)
