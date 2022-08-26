import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.qdist import QDistMixer
import torch as th
import numpy as np
from torch.optim import RMSprop
from torch.optim import Adam
from torch.distributions import Categorical
from torch.distributions import Beta
from torch.distributions import Dirichlet


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0


        self.mixer = QDistMixer(args)
        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.agent_taus = th.FloatTensor([((i+1)/ args.dist_N) for i in range(self.args.dist_N)]).view(1,1,args.dist_N,1,1).to(self.args.device)

        self.N = self.args.dist_N
        self.N_env = self.args.dist_N_env
        self.Np_env = self.args.dist_Np_env
        self.K_env = self.args.dist_K_env
        self.K = self.args.dist_K

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        mac_out = []
        mac_hidden_states = []

        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            mac_hidden_states.append(self.mac.hidden_states)


        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_hidden_states = th.stack(mac_hidden_states, dim=1)
        mac_hidden_states = mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1,2) #btav

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=4, index=actions.unsqueeze(2).repeat(1, 1, self.N, 1, 1)).squeeze(4)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_mac_hidden_states = []

        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_mac_hidden_states.append(self.target_mac.hidden_states)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time
        target_mac_hidden_states = th.stack(target_mac_hidden_states, dim=1)
        target_mac_hidden_states = target_mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1,2) #btav
        
        # Mask out unavailable actions
        # target_mac_out[avail_actions[:, :] == 0] = -9999999

        mac_out_detach = mac_out.clone().detach()
        mac_out_detach[avail_actions.unsqueeze(2).repeat(1, 1, self.N, 1, 1) == 0] = -9999999

        if self.args.risk_agent == 'seek':
            mac_out_detach = mac_out_detach[:, :, -1].unsqueeze(2)
        elif self.args.risk_agent == 'neutral':
            mac_out_detach = mac_out_detach[:, :, self.args.dist_N//2].unsqueeze(2)
        elif self.args.risk_agent == 'averse':
            mac_out_detach = mac_out_detach[:, :, 0].unsqueeze(2)        
        else:
            assert 0

        cur_max_actions = mac_out_detach.max(dim=-1, keepdim=True)[1]

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            # target_max_qvals = th.gather(target_mac_out[:, 1:], dim=-1, index=cur_max_actions[:,1:]).squeeze(-1)
            target_max_qvals = th.gather(target_mac_out[:, 1:], dim=-1, index=cur_max_actions[:,1:].repeat(1, 1, self.N, 1, 1)).squeeze(-1)

        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
        
        max_action_qvals = th.gather(mac_out[:, :-1], dim=-1, index=cur_max_actions[:,:-1].repeat(1, 1, self.N, 1, 1)).squeeze(-1)

        # Mix
        if self.mixer is not None:
            chosen_action_qvals_r, chosen_action_qvals2_r, taus = self.mixer(chosen_action_qvals, batch["state"][:, :-1], mac_hidden_states[:, :-1], mode = 'approx')
            chosen_action_qvals, _, taus = self.mixer(chosen_action_qvals, batch["state"][:, :-1], mac_hidden_states[:, :-1], mode = 'policy')
            max_action_qvals_r, max_action_qvals2, _ = self.mixer(max_action_qvals, batch["state"][:, :-1], mac_hidden_states[:, :-1], mode = 'approx')
            target_max_qvals, _, _ = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target_mac_hidden_states[:, 1:], mode = 'target')


        # exit()
        
        bs = chosen_action_qvals.size(0)
        ts = chosen_action_qvals.size(1)
        rewards = rewards.unsqueeze(2).unsqueeze(2)
        terminated = terminated.unsqueeze(2).unsqueeze(2)

        assert chosen_action_qvals_r.shape == (bs, ts, self.N, self.N_env, 1)
        assert chosen_action_qvals2_r.shape == (bs, ts, self.N, 1, 1)

        # v17
        # targets = (rewards + self.args.gamma * (1 - terminated) * target_max_qvals.min(dim = 2, keepdim = True)[0]).detach()
        targets = (rewards + self.args.gamma * (1 - terminated) * target_max_qvals.mean(dim = 2, keepdim = True)).detach()
        # targets = (rewards + self.args.gamma * (1 - terminated) * target_max_qvals).detach()

        #dim = 3은 환경의 fraction에 대한 평균
        targets2 = chosen_action_qvals_r.mean(dim = 3, keepdim = True)
        targets3 = max_action_qvals_r.mean(dim = 3, keepdim = True).detach()
        targets4 = max_action_qvals_r.mean(dim = 3, keepdim = True)

        # Td-error
        td_error = (chosen_action_qvals - targets.view(bs, ts, 1, 1, self.Np_env))
        weight = abs(taus -(td_error.detach() > 0).float())

        agent_taus = self.agent_taus.repeat(bs, ts, 1, 1, 1) #* 0 + 1

        #loss nopt
        td_error2 = chosen_action_qvals2_r - targets2
        weight2 = th.where(td_error2 > 0, 2 - 2 * agent_taus, th.ones_like(agent_taus))

        #loss ub
        td_error3 = chosen_action_qvals2_r - th.max(targets2, targets3)
        weight3 = th.where(td_error3 > 0, th.ones_like(agent_taus), th.zeros_like(agent_taus))
        
        #Loss opt
        td_error4 = max_action_qvals2 - targets4
        weight4 = 1

        mask = mask.unsqueeze(2).unsqueeze(2)
        mask1 = mask.expand_as(td_error)
        mask2 = mask.expand_as(td_error2)
        mask3 = mask.expand_as(td_error3)
        mask4 = mask.expand_as(td_error4)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask1
        masked_td_error2 = td_error2 * mask2
        masked_td_error3 = td_error3 * mask3
        masked_td_error4 = td_error4 * mask4

        huber_loss1 = th.where(masked_td_error.abs() <= 1, masked_td_error ** 2, 2 * masked_td_error.abs() - 1)
        huber_loss2 = masked_td_error2 ** 2
        huber_loss3 = masked_td_error3 ** 2
        huber_loss4 = masked_td_error4 ** 2

        # Normal L2 loss, take mean over actual data
        loss = 1 * (weight * (huber_loss1)).sum() / mask1.sum()
        loss += 1 * (weight2 * (huber_loss2)).sum() / mask2.sum()
        loss += 1 * (weight3 * (huber_loss3)).sum() / mask3.sum()
        loss += 3 * (weight4 * (huber_loss4)).sum() / mask4.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems1 = mask1.sum().item()
            mask_elems2 = mask2.sum().item()
            mask_elems3 = mask3.sum().item()



            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems1), t_env)
            self.logger.log_stat("td_error_abs2", (masked_td_error2.abs().sum().item()/mask_elems2), t_env)
            self.logger.log_stat("td_error_abs3", (masked_td_error3.abs().sum().item()/mask_elems3), t_env)

            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems1), t_env)
            self.logger.log_stat("target_mean", (targets * mask1).sum().item()/(mask_elems1), t_env)

            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
