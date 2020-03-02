# https://github.com/hunkim/ReinforcementZeroToAll
# 06_q_net_frozenlake.py
# convert to tensorflow 2.1?

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# Input and output size based on the Env
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

# These lines establish the feed-forward part of the network used to choose actions
W = tf.Variable(tf.random.uniform([input_size, output_size], 0, 0.01))  # weight


@tf.function
def q_pred(X):
    return tf.matmul(tf.cast(X, tf.float32), W)


@tf.function
def loss(_q_pred, Y):
    return tf.reduce_sum(tf.square(Y-_q_pred))


optimizer = tf.optimizers.SGD(learning_rate=learning_rate, decay=0.0, name='SGD')

# Set Q-learning related parameters
dis = 0.99
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []


def one_hot(x):
    return np.identity(16)[x:x + 1]


trainable_variables = [W]
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    e = 1. / ((i / 50) + 10)
    rAll = 0
    done = False
    local_loss = []

    # The Q-Network training
    while not done:
        # Choose an action by greedily (with e chance of rnadom action) from the Q-network
        Qs = np.array(q_pred(one_hot(s)))
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)

        # Get new state and reward from environment
        s1, reward, done, _ = env.step(a)
        if done:
            # Update Q, and no Qs+1, since it's a terminal state
            Qs[0, a] = reward

        else:
            # Obtain the Q_s1 values by feeding the new state through our network
            Qs1 = q_pred(one_hot(s1))
            # Update Q, and no Qs+1, since it's a terminal state
            Qs[0, a] = reward + dis * np.max(Qs1)

        # Train our network using target (Y) and predicted Q (Qpred)
        with tf.GradientTape() as g:
            _loss = loss(q_pred(one_hot(s)), Qs)

        gradients = g. gradient(_loss, trainable_variables)
        # print(gradients)
        # print(Qs)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        rAll += reward
        s = s1
    rList.append(rAll)

print('Percent of successful episodes : ' + str(sum(rList)/num_episodes) + '%')
plt.bar(range(len(rList)), rList, color='blue')
plt.show()

# percent of successful episodes : 0.413%