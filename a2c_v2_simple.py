import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env_name = 'CartPole-v0'

env = gym.make(env_name)
# env.seed(args.seed)
# torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

        # actor
        self.actor = nn.Linear(128, 2)

        # critic
        self.value = nn.Linear(128, 1)

        # action reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))

        action_prob = F.softmax(self.actor(x), dim=-1)

        state_values = self.value(x)

        return action_prob, state_values

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)

    m = Categorical(probs)

    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take
    return action.item()

def finish_episode():
    # Calculates actor and critic loss ad performs backprop

    R = 0

    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the dsicounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        policy_losses.append(-log_prob * advantage)

        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    optimizer.zero_grad()

    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10

    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0

        # Steps
        for t in range(1, 10000):
            action = select_action(state)

            state, reward, done, _ = env.step(action)

            # env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

main()        


