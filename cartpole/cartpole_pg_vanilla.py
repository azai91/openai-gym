import torch.nn.functional as F
import torch
import gym
import random
import numpy as np

from torch import nn
from torch.autograd import Variable
from torch import Tensor
import torch.optim as optim

gamma = 0.97

class PolicyGradient(nn.Module):
    def __init__(self):
        super(PolicyGradient,self).__init__()
        self.linear = nn.Linear(4,2)

    def forward(self, state):
        return F.softmax(self.linear(state),dim=1)


def run_episode(env, policy_gradient, pl_optimizer):

    observation = env.reset()
    rewards = []
    future_rewards = []
    loss = 0
    actions = []
    probs = []
    totalreward = 0

    pl_optimizer.zero_grad()

    for _ in range(200):
        obs_vector = Variable(Tensor(np.expand_dims(observation, axis=0)))
        prob = policy_gradient(obs_vector)
        action = 0 if random.uniform(0,1) < prob.data[0][0] else 1
        actionblank = np.zeros(2)
        actionblank[action] = 1

        # old_observation = observation
        observation, reward, done, info = env.step(action)
        totalreward += reward

        actions.append(actionblank)
        probs.append(prob)
        rewards.append(reward)

        if done:
            break

    discount = 1
    for i1 in range(len(rewards)):
        future_reward = 0
        for i2 in range(i1, len(rewards)):
            future_reward = discount * rewards[i2]
            discount *= gamma
        future_rewards.append(future_reward)

    # import ipdb
    # ipdb.set_trace()
    prob_vector = torch.cat(probs)
    action_vector = Variable(Tensor(actions)) # [N, 1]
    good_prob = (prob_vector * action_vector).sum(dim=1).unsqueeze(dim=1)
    reward_vector = Variable(Tensor(rewards).unsqueeze(1))
    loss = good_prob.log() * reward_vector # [N, 1]
    loss.backward()
    pl_optimizer.step()

    return totalreward

env = gym.make('CartPole-v0')
policy_grad = PolicyGradient()
pl_optimizer = optim.Adam(policy_grad.parameters(), lr=0.01)

rewards = []

for i in range(10000):
    reward = run_episode(env, policy_grad, pl_optimizer)
    rewards.append(reward)

rewards = np.array(rewards)
np.save('pg_vanilla.npy', rewards)



