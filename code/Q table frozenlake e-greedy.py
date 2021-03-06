# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.pjz9g59ap
# https://github.com/hunkim/ReinforcementZeroToAll
# 03_2_q_table_frozenlake_det.py
# convert to tensorflow 2.1

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register


register(  # https://stackoverflow.com/questions/52727233/how-can-i-register-a-custom-environment-in-openais-gym
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}  # Deterministic
)

env = gym.make('FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Discount factor
dis = .99

# Set learning parameters
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    e = 1. / ((i // 100) + 1)

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by e-greedy
        # Exploration

        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using decay rate
        Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print('Success rate : ' + str(sum(rList)/num_episodes))
print('Final Q-Table Values')
print(Q)
plt.bar(range(len(rList)), rList, color='blue')
plt.show()