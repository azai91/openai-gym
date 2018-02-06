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


class ValueGradient(nn.Module):
    def __init__(self):
        super(ValueGradient,self).__init__()
        self.linear1 = nn.Linear(4,10)
        self.linear2 = nn.Linear(10,1)

    def forward(self, state):
        h1 = F.relu(self.linear1(state))
        reward = self.linear2(h1)
        return reward


def run_episode(env, policy_gradient,value_gradient,pl_optimizer,vl_optimizer):

    observation = env.reset()
    transitions = []
    future_rewards = []
    actions = []
    probs = []
    pl_loss = 0
    totalreward = 0
    advantages = []
    pred_values = []

    pl_optimizer.zero_grad()
    vl_optimizer.zero_grad()

    for _ in range(200):
        obs_vector = Variable(Tensor(observation).unsqueeze(0))
        prob = policy_gradient(obs_vector)
        action = 0 if random.uniform(0,1) < prob.data[0][0] else 1
        actionblank = np.zeros(2)
        actionblank[action] = 1

        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        actions.append(actionblank)
        probs.append(prob)

        if done:
            break

    discount = 1
    for i1, trans in enumerate(transitions):
        obs, action, reward = trans

        future_reward = 0
        for i2 in range(i1, len(transitions)):
            future_reward = discount * transitions[i2][2]
            discount *= gamma
        future_rewards.append(future_reward)
        pred_value = value_gradient(Variable(Tensor(obs).unsqueeze(0)))
        pred_values.append(pred_value)

        advantages.append(future_reward - pred_value)



    vl_loss_func = nn.MSELoss()
    pred_value_vector = torch.cat(pred_values)
    vl_loss = vl_loss_func(pred_value_vector,Variable(Tensor(future_rewards).unsqueeze(1),requires_grad=False))
    vl_loss.backward()
    vl_optimizer.step()

    prob_vector = torch.cat(probs)
    action_vector = Variable(Tensor(actions), requires_grad=False) # [N, 1]
    good_prob = (prob_vector * action_vector).sum(dim=1)
    adv_vector = torch.cat(advantages)
    adv_vector.requires_grad = False
    pl_loss -= good_prob.log() * adv_vector # [N, 1]
    pl_loss.backward()
    pl_optimizer.step()

    return totalreward

env = gym.make('CartPole-v0')
policy_grad = PolicyGradient()
value_grad = ValueGradient()
pl_optimizer = optim.Adam(policy_grad.parameters(), lr=0.01)
vl_optimizer = optim.Adam(value_grad.parameters(), lr=0.1)

rewards = []

for i in range(10000):
    reward = run_episode(env,policy_grad,value_grad,pl_optimizer,vl_optimizer)
    print(reward)
    rewards.append(reward)

rewards = np.array(rewards)
np.save('pg_adv.npy', rewards)



