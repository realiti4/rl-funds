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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using: ', device)

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

from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

env_name = 'CartPole-v0'
env_name = 'BreakoutNoFrameskip-v4'

env    = make_atari(env_name)
env    = wrap_deepmind(env, frame_stack=True)
env    = wrap_pytorch(env)

# env = gym.make(env_name)
# env.seed(args.seed)
# torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),            
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # actor
        self.actor = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )

        # critic
        self.value = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # action reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x.unsqueeze(0)))
        x = x.view(x.size(0), -1)

        action_prob = F.softmax(self.actor(x), dim=-1)

        state_values = self.value(x)

        return action_prob, state_values

model = Policy().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().to(device)
    probs, state_value = model(state)

    m = Categorical(probs)

    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value.squeeze(0)))

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

        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))

    optimizer.zero_grad()

    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    loss.backward()
    optimizer.step()

    del model.rewards[:]
    del model.saved_actions[:]

import time

def main():
    running_reward = 10

    for i_episode in count(1):
        env.reset()
        state = env.reset()
        ep_reward = 0

        # Steps
        for t in range(1, 400000):
        # while True:
            action = select_action(state)

            state, reward, done, _ = env.step(action)
            reward = reward.item()

            # if i_episode > 20:
            #     env.render()

            env.render()
            # time.sleep(0.1)
            

            model.rewards.append(reward)
            ep_reward += reward
            if _['ale.lives'] == 0:
            # if done:
                print(ep_reward)
                break

            if t > 100000:
                print('alarm!')

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()
        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break

main()        


