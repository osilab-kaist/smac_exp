import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DMixer(nn.Module):
    def __init__(self, args):
        super(DMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.mixing_embed_dim
        self.hypernet_embed = args.hypernet_embed

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles
        self.n_agents = args.n_agents

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                            nn.ReLU(),
                            nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, target):
        batch_size = agent_qs.shape[0]
        episode_length = agent_qs.shape[1]
        if target:
            n_rnd_quantiles = self.n_target_quantiles
        else:
            n_rnd_quantiles = self.n_quantiles
        assert agent_qs.shape == (batch_size, episode_length, self.n_agents, n_rnd_quantiles)
        q_mixture = agent_qs.sum(dim=2, keepdim=True)
        assert q_mixture.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        q_vals_expected = agent_qs.mean(dim=3, keepdim=True)
        q_vals_sum = q_vals_expected.sum(dim=2, keepdim=True)
        assert q_vals_expected.shape == (batch_size, episode_length, self.n_agents, 1)
        assert q_vals_sum.shape == (batch_size, episode_length, 1, 1)

        # Factorization network
        q_joint_expected = self.forward_qmix(q_vals_expected, states)
        assert q_joint_expected.shape == (batch_size, episode_length, 1, 1)

        # Shape network
        q_vals_sum = q_vals_sum.expand(-1, -1, -1, n_rnd_quantiles)
        q_joint_expected = q_joint_expected.expand(-1, -1, -1, n_rnd_quantiles)
        assert q_vals_sum.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        assert q_joint_expected.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        q_joint = q_mixture - q_vals_sum + q_joint_expected
        assert q_joint.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        return q_joint

    def forward_qmix(self, agent_qs, states):
        batch_size = agent_qs.shape[0]
        episode_length = agent_qs.shape[1]
        assert agent_qs.shape == (batch_size, episode_length, self.n_agents, 1)
        bs = agent_qs.size(0)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        assert agent_qs.shape == (batch_size * episode_length, 1, self.n_agents)
        assert states.shape == (batch_size, episode_length, self.state_dim)
        states = states.reshape(-1, self.state_dim)
        assert states.shape == (batch_size * episode_length, self.state_dim)
        # First layer
        w1 = th.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        assert q_tot.shape == (batch_size, episode_length, 1)
        q_tot = q_tot.unsqueeze(3)
        assert q_tot.shape == (batch_size, episode_length, 1, 1)
        return q_tot
