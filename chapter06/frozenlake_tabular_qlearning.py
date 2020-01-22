# Author: bbrighttaer
# Project: DRL-lessons
# Date: 1/23/20
# Time: 12:05 AM
# File: frozenlake_tabular_qlearning.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

import gym
import numpy as np
import random

random.seed(0)
np.random.seed(0)

ENV_NAME = 'FrozenLake-v0'
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, s_prime):
        best_v, _ = self.best_value_and_action(s_prime)
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = (1 - ALPHA) * old_val + ALPHA * new_val

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            s_prime, reward, is_done, _ = env.step(action)
            # reduces number of iterations needed to solve problem
            self.value_update(state, action, reward, s_prime)
            total_reward += reward
            if is_done:
                break
            state = s_prime
        return total_reward


if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, s_prime = agent.sample_env()
        agent.value_update(s, a, r, s_prime)

        # Evaluation
        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        if reward > best_reward:
            print(f'Best reward updated {best_reward:.3f} -> {reward:.3f}')
            best_reward = reward
        if reward > .8:
            print(f'Solved in {iter_no} iterations!')
            break
