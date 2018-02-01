import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt
import tensorflow.contrib.eager as tfe


def policy_gradient():
    with tf.variable_scope('policy'):
        params = tf.get_variable('policy_parameters',[4,2])
        state = tf.placeholder('float',[None, 4])
        actions = tf.placeholder('float',[None,2])
        adv = tf.placeholder('float',[None, 1])
        linear = tf.matmul(state, params) # [None, 2]
        probs = tf.nn.softmax(linear) # [None, 2]
        good_probs = tf.reduce_sum(tf.mul(probs, actions), reduction_indices=[1]) # [None, 1]
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
            calc = tf.matmul(h1,w2)
            diffs = calc = newvals
            loss = tf.nn.l2_loss(diffs)
            optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
            return calc, state, newvals, optimizer, loss

    def run_episode(env, policy_grad, value_grad, sess):
        pl_calc, pl_state, pl_actions, pl_adv, pl_optimizer = policy_grad
        vl_calc, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
        observation = env.reset()
        totalreward = 0
        states, actions, advs = []



