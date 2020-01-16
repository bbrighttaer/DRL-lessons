# Author: bbrighttaer
# Project: DRL-lessons
# Date: 1/16/20
# Time: 11:17 PM
# File: cartpole.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import random
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.io import load_model, save_model
from utils.wrappers import DiscreteOneHotWrapper

HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, agent, batch_size, render=False):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        if render:
            env.render()
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(agent(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    disc_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)
    return elite_batch, train_obs, train_act, reward_bound


if __name__ == "__main__":
    random.seed(1234)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='train or eval')
    args = parser.parse_args()

    mode = args.mode

    env_name = 'FrozenLake-v0'
    env = DiscreteOneHotWrapper(gym.make(env_name))

    # env = gym.wrappers.Monitor(env, directory="mon", force=True)

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = Net(obs_size, HIDDEN_SIZE, n_actions)
    if mode == 'train':
        objective = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=agent.parameters(), lr=0.001)
    else:
        agent.load_state_dict(load_model('./', f'{env_name}.mod', dvc='cpu'))
        agent.eval()
    writer = SummaryWriter(log_dir='./logs', comment=env_name)

    # Interact with environment and gather training samples for training (if in training mode)
    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, agent, BATCH_SIZE, render=mode == 'eval')):
        # Select training samples that are above the specified threshold
        reward_m = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, acts, reward_b = filter_batch(batch + full_batch, PERCENTILE)

        """Model training"""
        if mode == 'train':
            if not full_batch:
                continue

            obs_v = torch.FloatTensor(obs)
            acts_v = torch.LongTensor(acts)
            full_batch = full_batch[-500:]

            # Clears gradients
            optimizer.zero_grad()

            # Forward pass
            action_scores_v = agent(obs_v)

            # Calculate loss
            loss_v = objective(action_scores_v, acts_v)

            # Backward pass
            loss_v.backward()

            # Update parameters
            optimizer.step()

            # Log to console and tensorboard
            print(f'{iter_no}: loss={loss_v:.3f}, reward_bound={reward_b:.1f}')
            writer.add_scalar("loss", loss_v.item(), iter_no)
            writer.add_scalar("reward_bound", reward_b, iter_no)
            writer.add_scalar("reward_mean", reward_m, iter_no)
        else:
            print(f'{iter_no}: reward_bound={reward_b:.1f}')
        if reward_m > 0.8:
            print("Solved!")
            if mode == 'train':
                save_model(agent, './', env_name)
            break
    writer.close()
    env.close()
