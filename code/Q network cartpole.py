# https://github.com/hunkim/ReinforcementZeroToAll
# 07_1_q_net_cartpole.py
# to tensorflow 2.1?

import numpy as np
import tensorflow as tf
from collections import deque
import gym
from time import sleep
env = gym.make('CartPole-v0')


# Constants defining our neural network
learning_rate = 1e-1
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# First layer of weights
initializer = tf.initializers.GlorotUniform()
W = tf.Variable(initializer(shape=(input_size, output_size)), name='W')


@tf.function
def q_pred(x): 
    return tf.matmul(tf.cast(x, tf.float32), W)


# Loss function
@tf.function
def loss(_q_pred, y):
    return tf.reduce_sum(tf.square(y - _q_pred))


# optimizer
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# Values for q learning
max_episodes = 5000
dis = 0.9
step_history = []

trainable_variables = [W]

for episode in range(max_episodes):
    e = 1. / ((episode / 10) + 1)
    step_count = 0
    state = env.reset()
    done = False
    
    # The Q-Network training
    while not done:
        step_count += 1
        x = np.reshape(state, [1, input_size])
        # Choose an action by greedily (with e chance of random action) from
        # the Q-network
        Q = np.array(tf.matmul(tf.cast(x, tf.float32), W))
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q)

        # Get new state and reward from environment
        next_state, reward, done, _ = env.step(action)
        if done:
            Q[0, action] = reward
        else:
            x_next = np.reshape(next_state, [1, input_size])
            Q_next = q_pred(tf.cast(x_next, tf.float32))
            Q[0, action] = reward + dis * np.max(Q_next)

        # Train our network using target and predicted Q values aon each episode
        with tf.GradientTape() as g:
            # loss = tf.reduce_sum(tf.square(Q - tf.matmul(tf.cast(x, tf.float32), W)))
            _loss = loss(q_pred(x), Q)

        gradients = g.gradient(_loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        state = next_state

    step_history.append(step_count)
    print('Episode: {} steps: {}'.format(episode, step_count))
    # If last 10's avg steps are 500, it's good enough
    if len(step_history) > 10 and np.mean(step_history[-10:]) > 500:
        break

# See our trained network in action
observation = env.reset()
reward_sum = 0
while True:
    env.render()
    sleep(0.03)
    x = np.reshape(observation, [1, input_size])
    Q = q_pred(x)
    action = np.argmax(Q)

    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        print('Total score: {}'.format(reward_sum))
        break


# score 34.0, 22.0
