import torch as th
import torch.nn as nn

class DDNMixer(nn.Module):
    def __init__(self, args):
        super(DDNMixer, self).__init__()
        self.args = args

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles
        self.n_agents = args.n_agents

    def forward(self, agent_qs, batch, target):
        batch_size = agent_qs.shape[0]
        episode_length = agent_qs.shape[1]
        if target:
            n_rnd_quantiles = self.n_target_quantiles
        else:
            n_rnd_quantiles = self.n_quantiles
        assert agent_qs.shape == (batch_size, episode_length, self.n_agents, n_rnd_quantiles)
        q_mixture = agent_qs.sum(dim=2, keepdim=True)
        assert q_mixture.shape == (batch_size, episode_length, 1, n_rnd_quantiles)
        return q_mixture