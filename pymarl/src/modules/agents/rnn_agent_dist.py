import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np


class RNNDistAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNDistAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions * self.args.dist_N)
        
        self.n_agents = self.args.n_agents

        self.Ws = th.FloatTensor([((i+1)/ args.dist_N) for i in range(self.args.dist_N)]).view(1,args.dist_N,1).to(self.args.device)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)

        q = q.view(inputs.size(0) // self.args.n_agents, self.n_agents, self.args.dist_N, -1).transpose(1, 2)

        return q, h
