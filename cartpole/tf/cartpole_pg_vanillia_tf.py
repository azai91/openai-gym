import tensorflow as tf
import numpy as np
import random
import gym

gamma = 0.97

def policy_gradient():
    with tf.variable_scope('policy'):
        params = tf.get_variable('policy_parameters',[4,2])
        state = tf.placeholder('float',[None, 4])
        actions = tf.placeholder('float',[None,2])
        reward = tf.placeholder('float',[None, 1])
        linear = tf.matmul(state, params) # [None, 2]
        probs = tf.nn.softmax(linear) # [None, 2]
        good_probs = tf.reduce_sum(tf.multiply(probs, actions), reduction_indices=[1]) # [None, 1]
        elg = tf.log(good_probs) * reward # pg method
        loss = -tf.reduce_sum(elg)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probs, state, actions, reward, optimizer


def run_episode(env, policy_grad, sess):
    pl_calc, pl_state, pl_actions, pl_reward, pl_optimizer = policy_grad
    observation = env.reset()
    totalreward = 0
    states, actions, rewards, transitions, update_vals = [],[],[],[],[]
    future_rewards = []

    # follow policy for episode
    for _ in range(200):
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calc, feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0,1) < probs[0][0] else 1 # take action
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)

        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        totalreward += reward

        if done:
            break


    for index in range(len(rewards)):
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
            future_reward += rewards[index] * decrease
            decrease = decrease * gamma
        future_rewards.append(future_reward)

    future_rewards_vector = np.expand_dims(future_rewards, axis=1) # [n,1]
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_reward: future_rewards_vector, pl_actions: actions})

    return totalreward

env = gym.make('CartPole-v0')
policy_grad = policy_gradient()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

rewards = []

for i in range(10000):
    reward = run_episode(env, policy_grad, sess)
    rewards.append(reward)

rewards = np.array(rewards)
np.save('pg_vanilla.npy', rewards)










