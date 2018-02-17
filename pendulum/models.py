from torch import nn
import torch.nn.functional as F
import torch

class Actor(nn.Module):
    def __init__(self, state_dim ,action_dim, action_lim, init_w=3e-3):
        super(Actor, self).__init__()
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)


    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = F.tanh(self.fc4(x))
        return action * self.action_lim


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Critic, self).__init__()
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        self.fcs1 = nn.Linear(state_dim,256)
        self.fcs2 = nn.Linear(256, 128)
        self.fca1 = nn.Linear(action_dim, 128)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2, a1), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x