import gym
from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import numpy as np

from replay_buffer import ReplayBuffer

from pendulum.models import Actor, Critic
from pendulum.utils import OrnsteinUhlenbeckActionNoise, copy_params

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
    count = 0

    for r in range(MAX_ITERATIONS):
        obs_vector = Variable(Tensor(observation).unsqueeze(0))
        # need to insert exploration
        orig_action = target_actor(obs_vector)
        action = orig_action.data.numpy()[0] + noise.sample()
        old_observation = observation
        observation, reward, done, info = env.step(action)
        totalreward += reward


        buffer.add((old_observation, action, reward, observation, orig_action))
        count += 1

        if len(buffer) > WARMUP_TIME:
            env.render()

            observations, actions, rewards, next_observations, orig_action = buffer.sample_batch(MINI_BATCH_SIZE)

            obs_vector = Variable(Tensor(observations))
            action_vector = Variable(Tensor(actions))
            next_obs_vector = Variable(Tensor(next_observations))

            # critic update

            next_actions = actor_network(next_obs_vector).detach()
            next_val_vector = torch.squeeze(target_critic(next_obs_vector, next_actions).detach())

            rewards_vector = Variable(Tensor(rewards)) + GAMMA * next_val_vector

            predicted_values_vector = critic_network(obs_vector, action_vector)
            critic_loss = F.mse_loss(predicted_values_vector, rewards_vector)
            critic_optimizer.zero_grad()

            critic_loss.backward()
            critic_optimizer.step()

            # actor update

            new_actions = actor_network(obs_vector)

            value_vector = critic_network(obs_vector, new_actions)
            actor_loss = -torch.mean(value_vector)

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            copy_params(target_actor, actor)
            copy_params(target_critic, critic)

        if count % MAX_STEPS_PER_EPISODE == 0:
            done = True

        if done:
            observation = env.reset()

    return totalreward


env = gym.make('Pendulum-v0')
S_DIM = env.observation_space.shape[0]
A_DIM = env.action_space.shape[0]
A_MAX = float(env.action_space.high[0])
BUFFER_SIZE = 1000000
MAX_ITERATIONS = 1000
GAMMA = 0.99
MINI_BATCH_SIZE = 128
WARMUP_TIME = 300
MAX_STEPS_PER_EPISODE = 200


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
    print('EPOCH', epoch)
    reward = train(env, actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, noise, replay_buffer)
