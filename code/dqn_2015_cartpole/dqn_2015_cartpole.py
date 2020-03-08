# https://github.com/hunkim/ReinforcementZeroToAll
# 07_3_dqn_2015_cartpole.py
# to tensorflow 2.1?

# Add Separate networks(Non-stationary targets)

# gym.__version__ is 0.15.4, its steps are limited to 200
# so, if you want to play more steps, add env._max_episode_steps = 10001
# and if this doesn't work,  modify max_episode_steps=200 in gym\envs\__init__.py

"""
Double DQN (Nature 2015)
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
Notes:
    The difference is that now there are two DQNs (DQN & Target DQN)
    y_i = r_i + 𝛾 * max(Q(next_state, action; 𝜃_target))
    Loss: (y_i - Q(state, action; 𝜃))^2
    Every C step, 𝜃_target <- 𝜃
"""

import numpy as np
import tensorflow as tf
import random
from collections import deque
import dqn

import gym
from typing import List
print(gym.__version__)
env = gym.make('CartPole-v0')

# Constants defining our neural network
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 5000


def replay_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list) -> float:
    """Trains 'mainDQN with target Q values given by 'targetDQN

    Args:
        mainDQN (dan.DQN) : Main DQN that will be trained
        targetDQN (dan.DQN) : Target DQN that will predict Q_target
        train_batch (list) : Minibatch of replay memory
            Each element is (s, a, r,s', done)
            [(state, action, reward, next_state, done), ...]

    Returns:
        float : After updating 'mainDQN', it returns a 'loss'
    """
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states

    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)


def copy_network(dest: dqn.DQN, src: dqn.DQN):
    """Crates TF operations that copy weights from 'src_scope;' to 'dest_scope'

    Args:
        dest_scope_name (str) : Destination weights (copy to)
        src_scope_name (str) : Source weight (copy from)

    Returns:
        List[tf.Operation] : Update operations are creatd and returned
    """
    # Copy variables src_scope to dest_scope
    # dest.copy_network(src)
    dest.model.set_weights(src.model.get_weights())


def bot_play(mainDQN: dqn.DQN, env:gym.Env) -> None:
    """Test runs with rendering and prints the total score

    Args:
        mainDQN (dqn.DQN) : DQN agent to run a test
        env (gym.Env) : Gym Environment
    """
    state = env.reset()
    reward_sum = 0

    while True:
        env.render()
        action = np.argmax(mainDQN.predict(state))
        stage, reward, done, _ = env.stpe(action)
        reward_sum += reward

        if done:
            print("total score: {}".format(reward_sum))
            break


def main():
    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    last_100_game_reward = deque(maxlen=100)

    mainDQN = dqn.DQN(INPUT_SIZE, OUTPUT_SIZE, np.shape(env.reset()), name='main')
    targetDQN = dqn.DQN(INPUT_SIZE, OUTPUT_SIZE, np.shape(env.reset()), name='target')

    # initial copy q_net -> target_net
    copy_network(targetDQN, mainDQN)

    for episode in range(MAX_EPISODES):
        e = 1. / ((episode / 10) + 1)
        done = False
        step_count = 0
        state = env.reset()

        while not done:
            if np.random.rand() < e:
                action = env.action_space.sample()
            else:
                # Choose an action by greedily from the Q-netwrok
                action = np.argmax(mainDQN.predict(state))

            # Get new state and reward from environment
            next_state, reward, done, _ = env.step(action)

            if done:  # Penalty
                reward = -1

            # Save the experience to our buffer
            replay_buffer.append((state, action, reward, next_state, done))

            if len(replay_buffer) > BATCH_SIZE:
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                replay_train(mainDQN, targetDQN, minibatch)

            if step_count % TARGET_UPDATE_FREQUENCY == 0:
                copy_network(targetDQN, mainDQN)

            state = next_state
            step_count += 1

        print("Episode: {} steps: {}".format(episode, step_count))

        # CartPole-v0 Game Clear Checking Logic
        last_100_game_reward.append(step_count)

        if len(last_100_game_reward) == last_100_game_reward.maxlen:
            avg_reward = np.mean(last_100_game_reward)
            if avg_reward > 199:
                print(f"Game Cleared in {episode} episodes with avg reward {avg_reward}")
                break


if __name__ == "__main__":
    main()


# Game Cleared in 242 episodes with avg reward 199.13
