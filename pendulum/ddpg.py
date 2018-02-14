import gym
from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import numpy as np

from replay_buffer import ReplayBuffer


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

def copy_params(target_network, source_network):
    for target_params, source_params in zip(target_network.parameters(), source_network.parameters()):
        target_params.data.copy_(source_params.data)

def soft_copy_params(target_network, source_network, tau=0.001):
    for target_param, source_params in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + source_params.data * tau
        )


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, state_dim ,action_dim, action_lim, init_w=3e-3):
        super(Actor, self).__init__()
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        self.action_lim = action_lim

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)
        # self.init_weights(init_w)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data.uniform_(-init_w, init_w)

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

MAX_STEPS_PER_EPISODE = 1000
GAMMA = 0.99
MINI_BATCH_SIZE = 128

def train(
        env,
        actor_network,
        critic_network,
        target_actor,
        target_critic,
        actor_optimizer,
        critic_optimizer,
        noise,
        buffer
    ):
    observation = env.reset()
    totalreward = 0


    for r in range(MAX_STEPS_PER_EPISODE):
        # env.render()
        obs_vector = Variable(Tensor(observation).unsqueeze(0))
        # need to insert exploration
        action = actor_network(obs_vector)
        action = action.data.numpy()[0] + noise.sample()
        old_observation = observation
        observation, reward, done, info = env.step(action)
        totalreward += reward

        buffer.add((old_observation, action, reward, observation))

        if len(buffer) > MINI_BATCH_SIZE:

            observations, actions, rewards, next_observations = buffer.sample_batch(MINI_BATCH_SIZE)

            obs_vector = Variable(Tensor(observations))
            action_vector = Variable(Tensor(actions))
            next_obs_vector = Variable(Tensor(next_observations))


            # critic update

            next_actions = target_actor(next_obs_vector).detach()
            next_val_vector = torch.squeeze(target_critic(next_obs_vector, next_actions).detach())

            rewards_vector = Variable(Tensor(rewards)) + GAMMA * next_val_vector

            predicted_values_vector = critic_network(obs_vector, action_vector)
            critic_loss = F.smooth_l1_loss(predicted_values_vector, rewards_vector)
            critic_optimizer.zero_grad()

            critic_loss.backward(retain_graph=True)
            critic_optimizer.step()

            # actor update

            new_actions = actor_network(obs_vector)

            value_vector = critic_network(obs_vector, new_actions)
            actor_loss = -torch.sum(value_vector)

            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optimizer.step()

            soft_copy_params(target_actor, actor)
            soft_copy_params(target_critic, critic)
            print(critic_loss.data[0] / MINI_BATCH_SIZE, actor_loss.data[0] / MINI_BATCH_SIZE)

        if done:
            break


    return totalreward





env = gym.make('Pendulum-v0')
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = float(env.action_space.high[0])
BUFFER_SIZE = 1000000

actor = Actor(S_DIM, A_DIM, A_MAX)
critic = Critic(S_DIM, A_DIM)
target_actor = Actor(S_DIM, A_DIM, A_MAX)
target_critic = Critic(S_DIM, A_DIM)

copy_params(target_actor, actor)
copy_params(target_critic, critic)

actor_optimizer = optim.Adam(actor.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic.parameters(), lr=0.001)
noise = OrnsteinUhlenbeckActionNoise(A_DIM, A_MAX)

replay_buffer = ReplayBuffer(BUFFER_SIZE)
EPOCHS = 2000

for epoch in range(EPOCHS):
    reward = train(env, actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, noise, replay_buffer)
    # print(reward)
