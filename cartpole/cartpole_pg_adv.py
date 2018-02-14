import torch.nn.functional as F
import torch
import gym
import random
import numpy as np

from torch import nn
from torch.autograd import Variable
from torch import Tensor
import torch.optim as optim
from torch.distributions import Categorical

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


def run_episode(env, policy_gradient, value_gradient, pl_optimizer, vl_optimizer):

    observation = env.reset()
    transitions = []
    future_rewards = []
    log_prob_actions = []
    totalreward = 0
    advantages = []
    pred_values = []

    pl_optimizer.zero_grad()
    vl_optimizer.zero_grad()

    for _ in range(200):
        obs_vector = Variable(Tensor(observation).unsqueeze(0))
        prob = policy_gradient(obs_vector)
        m = Categorical(prob)
        action = m.sample()

        old_observation = observation
        observation, reward, done, info = env.step(action.data[0])
        transitions.append((old_observation, None, reward))
        totalreward += reward

        log_prob_actions.append(m.log_prob(action))

        if done:
            break

    for i1, trans in enumerate(transitions):
        obs, _, _ = trans

        future_reward = 0
        discount = 1
        for i2 in range(i1, len(transitions)):
            reward = transitions[i2][2]
            future_reward += discount * reward
            discount *= gamma
        future_rewards.append(future_reward)
        pred_value = value_gradient(Variable(Tensor(obs).unsqueeze(0)))
        pred_values.append(pred_value)

        advantages.append(future_reward - pred_value.data[0][0])

    vl_loss_func = nn.MSELoss()

    pred_value_vector = torch.cat(pred_values) # [N]
    future_values_vector = Variable(Tensor(future_rewards).unsqueeze(1))
    vl_loss = vl_loss_func(pred_value_vector,future_values_vector)
    vl_loss.backward()
    vl_optimizer.step()

    actions_vector = torch.cat(log_prob_actions) # [N]
    adv_vector = Variable(Tensor(advantages)) # [N]
    pl_loss = -(actions_vector * adv_vector).sum()
    pl_loss.backward()
    pl_optimizer.step()

    return totalreward, advantages, vl_loss

env = gym.make('CartPole-v0')
policy_grad = PolicyGradient()
value_grad = ValueGradient()
pl_optimizer = optim.Adam(policy_grad.parameters(), lr=0.01)
vl_optimizer = optim.Adam(value_grad.parameters(), lr=0.1)

rewards = []

for i in range(1000):
    reward, advantages, vl_loss = run_episode(env,policy_grad,value_grad,pl_optimizer,vl_optimizer)
    rewards.append(reward)

rewards = np.array(rewards)
np.save('pg_adv_refactor.npy', rewards)



