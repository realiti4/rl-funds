import math
import random, os

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

from collections import deque

# num_envs = 1
env_name = 'BreakoutNoFrameskip-v4'
# env_name = 'PongNoFrameskip-v4'


# baselines' env.make
from envs_sb import make_vec_envs

# num_steps = 5
# num_processes = 16

# envs = make_vec_envs(env_name, 1, num_processes, 0.99, '/tmp/gym/', device, False)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Model(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Model, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init__ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), gain=0.01)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.actor_linear = init__(nn.Linear(hidden_size, 4))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, x):
        x = self.main(x / 255.0)

        return self.critic_linear(x), x

    def act(self, state):
        value, actor_features = self.forward(state)
        probs = self.actor_linear(actor_features)
        dist = Categorical(logits=probs)     # you are using 'Probs', example uses 'logits'
        
        action = dist.sample().unsqueeze(-1)
        # TODO look what does sum() do here..
        action_log_probs = dist.log_prob(action.squeeze(-1)).view(action.size(0), -1).sum(-1).unsqueeze(-1)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def evaluate_actions(self, inputs, action):
        # action = torch.cat(action).to(device)
        value, actor_features = self.forward(inputs.to(device))
        probs = self.actor_linear(actor_features)
        dist = Categorical(logits=probs)

        action_log_probs = dist.log_prob(action.squeeze(-1)).view(action.size(0), -1).sum(-1).unsqueeze(-1)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

class memory():
    def __init__(self):
        self.state = torch.zeros([num_steps+1, 1, 4, 84, 84]).to(device)

    def update(self, step, state, reward=None):
        if type(state).__module__ == np.__name__:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.state[step] = state

    def reset(self):
        self.state = torch.zeros([num_steps+1, 4, 84, 84]).to(device)

# memory = memory()


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step].to(device) + gamma * R * masks[step].to(device)
        returns.insert(0, R)
    return returns

def main(): 
    # torch.set_num_threads(1)

    num_steps = 5
    num_processes = 8

    envs = make_vec_envs(env_name, 1, num_processes, 0.99, '/tmp/gym/', device, False)


    state = envs.reset()

    model = Model(4, 512).to(device)
    optimizer = optim.RMSprop(model.parameters(), 0.0007, eps=1e-05, alpha=0.99)

    episode_rewards = deque(maxlen=10)

    import time
    start = time.time()

    epoch = 1
    load = False
    save = False
    if load:
        checkpoint = torch.load(f'checkpoints/a2c_breakout/checkpoint.pth')
        model.load_state_dict(checkpoint['current_state_dict'])
        optimizer.load_state_dict(checkpoint['policy_optim_dict'])
        model.train()
        epoch = checkpoint['epoch']

    for i_episode in range(epoch+1, 1000000):    

        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0

        # test1
        states = torch.zeros([num_steps, num_processes, 4, 84, 84])
        actions = []

        for step in range(num_steps):
            # test1
            states[step] = state

            if i_episode >= 72000:
                envs.render()
                time.sleep(0.04)

            # Sample
            with torch.no_grad():
                value, action, action_log_prob = model.act(state)

            # Observe reward and next state
            next_state, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    # print(info['episode']['r'])
                    

            # If done then clean the history of observations.
            mask = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])


            rewards.append(reward)
            masks.append(mask)
            # test1
            actions.append(action)

            state = next_state

        
        with torch.no_grad():
            next_value, _ = model(state)
        
        returns = compute_returns(next_value, rewards, masks)
        returns = torch.stack(returns)

        actions = torch.stack(actions)
            
        # values, action_log_probs, dist_entropy = model.evaluate_actions(states, actions)
        values, action_log_probs, dist_entropy = model.evaluate_actions(states.view(-1, 4, 84, 84), actions.view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)
        # returns = torch.cat(returns).view(5, 1, 1)

        advantages = returns - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        optimizer.zero_grad()

        value_loss_coef = 0.5
        entropy_coef = 0.01

        (value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef).backward()

        max_grad_norm = 0.5
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()                           

        if i_episode % 400 == 0:
            time_log = round((time.time() - start), 2)
            print(f'mean: {round(sum(episode_rewards) / len(episode_rewards))}, max: {max(episode_rewards)}, time: {time_log} secs')

            start = time.time()

        # Save
        if i_episode % 10000 == 0:
            if save:
                # save the model
                checkpoint_path = 'checkpoints/a2c_breakout/'
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                torch.save({
                    'epoch': i_episode,
                    'current_state_dict': model.state_dict(),
                    'policy_optim_dict': optimizer.state_dict()}, checkpoint_path + f'checkpoint.pth')
                print(f'saved heeey. episode: {i_episode}')

if __name__ == '__main__':
    main()