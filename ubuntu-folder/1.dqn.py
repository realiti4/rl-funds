import os
import math, random
import time

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)

env_id = "CartPole-v1"
# env_id = 'MountainCar-v0'
env = gym.make(env_id)

eps_start = 1.0
eps_final = 0.01
eps_decay = 500


# TODO improve select action
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_treshold = eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)

    state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    
    steps_done += 1
    if sample > eps_treshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].item()
    else:
        # return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)
        return random.randrange(env.action_space.n)


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)

            with torch.no_grad():
                q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(env.action_space.n)
        return action


model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    
optimizer = optim.Adam(model.parameters())

replay_buffer = ReplayBuffer(10000)        


def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.from_numpy(np.float32(state)).to(device)
    next_state = torch.from_numpy(np.float32(next_state)).to(device)    
    action     = torch.LongTensor(action).to(device)
    reward     = torch.FloatTensor(reward).to(device)
    done       = torch.FloatTensor(done).to(device)

    q_values      = model(state)
    next_q_values = model(next_state)
    # next_q_state_values = target_net(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    # next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss


def plot(frame_idx, rewards, losses):
    # clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


num_frames = 10000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

start = time.time()

state = env.reset()
for frame_idx in range(1, num_frames + 1):
    # epsilon = epsilon_by_frame(frame_idx)
    action = select_action(state)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    # if frame_idx > num_frames-1000: env.render()

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())
        
    if frame_idx % 200 == 0:
        print(np.mean(all_rewards[-10:]))
        # plot(frame_idx, all_rewards, losses)
        

env.close()
print(f'It took {time.time()-start} secs using {device}')


# Atari part
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch

env_id = "PongNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env)
env    = wrap_pytorch(env)


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            # nn.Conv2d(84, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def feature_size(self):
        # return self.features(autograd.Variable(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
            q_value = self.forward(state)
            # action  = q_value.max(1)[1].data[0]
            action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(env.action_space.n)
        return action


model = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)

   
optimizer = optim.Adam(model.parameters(), lr=0.00001)

replay_initial = 10000
replay_buffer = ReplayBuffer(100000)        


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)


num_frames = 1400000
batch_size = 64
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

start = time.time()

state = env.reset()
for frame_idx in range(1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = model.act(state, epsilon)
    # action = select_action(state)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward

    if frame_idx > 500010: env.render()
    
    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    if len(replay_buffer) > replay_initial:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())
        
    if frame_idx % 100000 == 0:
        # save the model
        checkpoint_path = 'checkpoints/pytorch/rl/dqn_1/'
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save({
            'epoch': frame_idx,
            'policy_state_dict': model.state_dict(),
            'policy_optim_dict': optimizer.state_dict()}, checkpoint_path + f'checkpoint.pth')
        print(f'saved heeey')    

        print(f'It took: {time.time()-start} secs')
        # plot(frame_idx, all_rewards, losses)
        start = time.time()