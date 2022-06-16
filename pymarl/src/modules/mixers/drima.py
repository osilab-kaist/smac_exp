import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DrimaMixer(nn.Module):
    def __init__(self, args):
        super(DrimaMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))



        self.embed_dim = args.mixing_embed_dim
        self.hidden_dim = 64

        self.weight1 = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.n_agents * self.args.dist_N))

        self.weight2 = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.n_agents * self.args.dist_N))

        self.V = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.args.dist_N))

        self.n_cos = self.args.n_cos
        self.pis = th.FloatTensor([np.pi*i for i in range(self.n_cos)]).view(1,1,1,self.n_cos).to(self.args.device)
        self.fc_cos = nn.Linear(self.n_cos, self.hidden_dim)


        self.Qjt = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.hidden_dim, 1))

        self.Qjt_embed = nn.Sequential(nn.Linear(self.state_dim + self.args.dist_N * self.n_agents, self.hidden_dim),
                               nn.ReLU(),
                               nn.Linear(self.hidden_dim, self.hidden_dim),
                               nn.ReLU())


    def forward(self, agent_qs, states, hidden_states, mode = None):
        bs = agent_qs.shape[0]
        ts = agent_qs.shape[1]
        N = agent_qs.shape[2]

        assert agent_qs.shape == (bs, ts, N, self.n_agents)
        assert states.shape == (bs, ts, self.state_dim)
        assert hidden_states.shape == (bs, ts, self.n_agents, self.args.rnn_hidden_dim)

        if mode == 'policy':
            N_env = self.args.dist_N_env
        elif mode == 'target':
            N_env = self.args.dist_Np_env
        elif mode == 'approx':
            N_env = self.args.dist_K_env
        else:
            assert 0

        taus_env = th.rand(bs, ts, N_env, 1).to(self.args.device)

        if mode == 'approx':
            if self.args.risk_env == 'neutral':
                taus_env = taus_env
            elif self.args.risk_env == 'averse':
                taus_env = taus_env * 0.25
            elif self.args.risk_env == 'seek':
                taus_env = taus_env * 0.25 + 0.75
            else:
                assert 0
                
        cos_embed_env = F.relu(self.fc_cos(th.cos(taus_env * self.pis)))
        agent_qs_e = agent_qs.view(bs, ts, 1, -1) #[32, 120, 1, 10 x 5]
        cat_state_qs = th.cat([agent_qs_e, states.unsqueeze(2)], dim = -1)

        Qjt_embed = self.Qjt_embed(cat_state_qs)
        assert Qjt_embed.shape == (bs, ts, 1, self.hidden_dim)

        Qjt = self.Qjt(Qjt_embed * cos_embed_env)

        v = self.V(states)
        w1 = th.abs(self.weight1(states)).reshape((bs, ts, N, self.n_agents))
        w2 = th.abs(self.weight2(states)).reshape((bs, ts, N, self.n_agents))

        q_tot = Qjt.reshape(bs, ts, 1, N_env, 1) + (w1 * agent_qs).sum(dim = -1).reshape(bs, ts, N, 1, 1) + v.reshape(bs, ts, N, 1, 1)
        q_tot2 = (w2 * agent_qs).sum(dim = -1).reshape(bs, ts, N, 1, 1) + v.reshape(bs, ts, N, 1, 1)

        return q_tot, q_tot2, taus_env.reshape(bs, ts, 1, N_env, 1)


