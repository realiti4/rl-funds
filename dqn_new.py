import math, random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using: ', device)

env = gym.envs.make("MountainCar-v0")


# Function Approximation
import sys
import itertools
from lib import plotting

# Estimator
class Estimator(nn.Module):
    def __init__(self):
        super(Estimator, self).__init__()

        self.dense1 = nn.Linear(2, 128)
        self.dense2 = nn.Linear(128, 3)

    def forward(self, state):
        output = self.dense1(state)
        return self.dense2(output)

    def predict(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            output = self.forward(state)
            return output.numpy()
    
print(env.action_space.n)

def make_epsilon_greedy_policy(estimator, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        # q_values = estimator.predict(observation)

        with torch.no_grad():
            observation = torch.from_numpy(observation).float()
            q_values = estimator(observation)
            q_values = q_values.numpy()

        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def q_learning(env, estimator, num_episodes, discount_factor=1.0, epsilon=0.1, epsilon_decay=1.0):
    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))   

    for i_episode in range(num_episodes):
        # The policy we're following
        policy = make_epsilon_greedy_policy(
            estimator, epsilon * epsilon_decay**i_episode, env.action_space.n)

        # Debugging
        last_reward = stats.episode_rewards[i_episode - 1]
        sys.stdout.flush()

        state = env.reset()

        next_action = None

        for t in itertools.count():
            if next_action is None:
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            else:
                action = next_action
            
            next_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # TD Update
            with torch.no_grad():
                q_values_next = estimator(torch.from_numpy(next_state).float())
                q_values_next = q_values_next.numpy()

            td_target = reward + discount_factor * np.max(q_values_next)

            if done:
                break

            

            q_value = estimator(torch.from_numpy(state).float())
            # loss = torch.tensor(td_target).float() - q_value[action]
            loss = criterion(torch.tensor(td_target).float(), q_value[action])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state


    return stats

criterion = nn.MSELoss()
estimator = Estimator()
optimizer = optim.Adam(estimator.parameters())

stats = q_learning(env, estimator, 100, epsilon=0.0)

plotting.plot_cost_to_go_mountain_car(env, estimator)
plotting.plot_episode_stats(stats, smoothing_window=25)