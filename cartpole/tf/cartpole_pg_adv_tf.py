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
        adv = tf.placeholder('float',[None, 1])
        linear = tf.matmul(state, params) # [None, 2]
        probs = tf.nn.softmax(linear) # [None, 2]
        good_probs = tf.reduce_sum(tf.multiply(probs, actions), reduction_indices=[1]) # [None, 1]
        elg = tf.log(good_probs) * adv # pg method
        loss = -tf.reduce_sum(elg)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return probs, state, actions, adv, optimizer


def value_gradient():
    with tf.variable_scope('value'):
        state = tf.placeholder('float',[None,4])
        newvals = tf.placeholder('float',[None,1])
        w1 = tf.get_variable('w1',[4,10])
        b1 = tf.get_variable('b1',[10])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable('w2',[10,1])
        b2 = tf.get_variable('b2',[1])
        calc = tf.matmul(h1,w2) + b2 # [1]
        diffs = calc - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calc, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess):
    pl_calc, pl_state, pl_actions, pl_adv, pl_optimizer = policy_grad
    vl_calc, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states, actions, advs, transitions, update_vals = [],[],[],[],[]

    # follow policy for episode
    for _ in range(200):
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(pl_calc, feed_dict={pl_state: obs_vector})
        action = 0 if random.uniform(0,1) < probs[0][0] else 1 # take action
        states.append(observation)
        actionblank = np.zeros(2)
        actionblank[action] = 1
        actions.append(actionblank)

        old_observation = observation
        observation, reward, done, info = env.step(action)
        env.render()
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break

    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * gamma
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calc, feed_dict={vl_state: obs_vector})[0][0]

        advs.append(future_reward - currentval)
        update_vals.append(future_reward)

    update_vals_vector = np.expand_dims(update_vals, axis=1) # [n,1]
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    advs_vector = np.expand_dims(advs, axis=1) # [n,1]
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_adv: advs_vector, pl_actions: actions})

    return totalreward

env = gym.make('CartPole-v0')
policy_grad = policy_gradient()
value_grad = value_gradient()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

rewards = []

for i in range(10000):
    reward = run_episode(env, policy_grad, value_grad, sess)
    rewards.append(reward)

# rewards = np.array(rewards)
# np.save('pg_adv.npy', rewards)









