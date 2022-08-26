import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.maddpg import MADDPGCritic
import torch as th
from torch.optim import RMSprop, Adam


class MADDPGLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.target_mac = copy.deepcopy(self.mac)
        self.agent_params = list(mac.parameters())

        self.critic = MADDPGCritic(scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.agent_optimiser = RMSprop(params=self.agent_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        if getattr(self.args, "optimizer", "rmsprop") == "rmsprop":
            self.critic_optimiser = RMSprop(params=self.critic_params, lr=args.critic_lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif getattr(self.args, "optimizer", "rmsprop") == "adam":
            self.critic_optimiser = Adam(params=self.critic_params, lr=args.critic_lr, eps=getattr(args, "optimizer_epsilon", 10E-8))
        else:
            raise Exception("unknown optimizer {}".format(getattr(self.args, "optimizer", "rmsprop")))

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # Train the critic batched
        target_actions = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_target_outs = self.target_mac.select_actions(batch, t_ep=t, t_env=None, test_mode=True,
                                                               critic=self.target_critic, target_mac=True)
            target_actions.append(agent_target_outs)
        target_actions = th.stack(target_actions, dim=1)  # Concat over time

        q_taken = []
        for t in range(batch.max_seq_length - 1):
            inputs = self._build_inputs(batch, t=t)
            critic_out, _ = self.critic(inputs, actions[:, t:t+1].detach())
            critic_out = critic_out.view(batch.batch_size, -1, 1)
            q_taken.append(critic_out)
        q_taken = th.stack(q_taken, dim=1)

        target_vals = []
        for t in range(1, batch.max_seq_length):
            target_inputs = self._build_inputs(batch, t=t)
            target_critic_out, _ = self.target_critic(target_inputs, target_actions[:, t:t+1].detach())
            target_critic_out = target_critic_out.view(batch.batch_size, -1, 1)
            target_vals.append(target_critic_out)
        target_vals = th.stack(target_vals, dim=1)

        q_taken = q_taken.view(batch.batch_size, -1, 1)
        target_vals = target_vals.view(batch.batch_size, -1, 1)
        targets = rewards.expand_as(target_vals) + self.args.gamma * (1 - terminated.expand_as(target_vals)) * target_vals

        td_error = (q_taken - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        critic_grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        mac_out = []
        chosen_action_qvals = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.mac.forward(batch, t=t, select_actions=True)["actions"].view(batch.batch_size,
                                                                                           self.n_agents,
                                                                                           self.n_actions)

            for idx in range(self.n_agents):
                tem_joint_act = actions[:, t:t+1].detach().clone().view(batch.batch_size, -1, self.n_actions)
                tem_joint_act[:, idx] = agent_outs[:, idx]
                q, _ = self.critic(self._build_inputs(batch, t=t), tem_joint_act)
                chosen_action_qvals.append(q.view(batch.batch_size, -1, 1))
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)
        chosen_action_qvals = th.stack(chosen_action_qvals, dim=1)
        pi = mac_out

        # Compute the actor loss
        pg_loss = -chosen_action_qvals.mean() + (pi**2).mean() * 1e-3

        # Optimise agents
        self.agent_optimiser.zero_grad()
        pg_loss.backward()
        agent_grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if getattr(self.args, "target_update_mode", "hard") == "hard":
            self._update_targets()
        elif getattr(self.args, "target_update_mode", "hard") in ["soft", "exponential_moving_average"]:
            self._update_targets_soft(tau=getattr(self.args, "target_update_tau", 0.001))
        else:
            raise Exception(
                "unknown target update mode: {}!".format(getattr(self.args, "target_update_mode", "hard")))

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("critic_loss", loss.item(), t_env)
            self.logger.log_stat("critic_grad_norm", critic_grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("q_taken_mean", (q_taken * mask).sum().item() / mask_elems, t_env)
            self.logger.log_stat("target_mean", targets.sum().item() / mask_elems, t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", agent_grad_norm, t_env)
            self.log_stats_t = t_env

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        if self.args.verbose:
            self.logger.console_logger.info("Updated all target networks (soft update tau={})".format(tau))

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = []

        # The centralized critic takes the state input, not observation
        inputs.append(batch["state"][:, t])

        if self.args.recurrent_critic:
            if self.args.obs_last_action:
                if t == 0:
                    inputs.append(th.zeros_like(batch["actions"][:, t]))
                else:
                    inputs.append(batch["actions"][:, t - 1])

        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        return inputs

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.logger.console_logger.info("Updated all target networks")

    def cuda(self, device="cuda:0"):
        self.mac.cuda(device=device)
        self.target_mac.cuda(device=device)
        self.critic.cuda(device=device)
        self.target_critic.cuda(device=device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.agent_optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        self.agent_optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))