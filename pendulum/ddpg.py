import gym
from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import numpy as np


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, action_max, mu = 0, theta = 0.15, sigma = 0.2):
        self.action_dim = action_dim
        self.action_max = action_max
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X * self.action_max

class Actor(nn.Module):
    def __init__(self, state_dim ,action_dim, action_lim):
        super(Actor, self).__init__()
        # self.state_dim = state_dim
        # self.action_dim = action_dim
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
    def __init__(self, state_dim, action_dim):
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

MAX_STEPS = 100
GAMMA = 0.5

def train(env, actor_network, critic_network, actor_optimizer, critic_optimizer, noise):
    observation = env.reset()
    transitions = []
    future_rewards = []
    predicted_values = []
    states = []
    actions = []
    totalreward = 0

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()

    for r in range(MAX_STEPS):
        # env.render()
        obs_vector = Variable(Tensor(observation).unsqueeze(0))
        # need to insert exploration
        action = actor_network(obs_vector)
        actions.append(action)
        action = action + Variable(Tensor(noise.sample()))
        old_observation = observation
        observation, reward, done, info = env.step(action.data[0])
        totalreward += reward
        transitions.append((old_observation, action, reward))
        states.append(old_observation)
        if done:
            break

    for i1 in range(len(transitions)):
        observation, action, reward = transitions[i1]
        discount = 1
        future_reward = 0
        for i2 in range(i1, len(transitions)):
            future_reward += transitions[i2][2] * discount
            discount *= GAMMA
        future_rewards.append(future_reward)
        pred_value = critic_network(Variable(Tensor(observation).unsqueeze(0)), action.detach())
        predicted_values.append(pred_value)

    pred_vector = torch.cat(predicted_values)
    future_rewards_vector = Variable(Tensor(future_rewards).unsqueeze(1))
    # import ipdb; ipdb.set_trace()
    critic_loss = F.smooth_l1_loss(pred_vector, future_rewards_vector)
    critic_loss.backward()
    critic_optimizer.step()

    action_vector = torch.cat(actions)
    value_vector = critic_network(Variable(Tensor(states)), action_vector)
    import ipdb; ipdb.set_trace()
    actor_loss = -torch.sum(value_vector)
    actor_loss.backward()
    actor_optimizer.step()
    print(critic_loss.data[0], actor_loss.data[0])

    return totalreward





env = gym.make('Pendulum-v0')
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = float(env.action_space.high[0])

actor = Actor(S_DIM, A_DIM, A_MAX)
critic = Critic(S_DIM, A_DIM)
actor_optimizer = optim.Adam(actor.parameters(), lr=0.01)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.1)
noise = OrnsteinUhlenbeckActionNoise(A_DIM, A_MAX)

EPOCHS = 200

for epoch in range(EPOCHS):
    reward = train(env, actor, critic, actor_optimizer, critic_optimizer, noise)
