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

# from IPython.display import clear_output
import matplotlib.pyplot as plt

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

env_id = "MountainCar-v0"
env = gym.make(env_id)

eps_start = 1.0
eps_final = 0.01
eps_decay = 5000

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_treshold = eps_final + (eps_start - eps_final) * math.exp(-1. * frame_idx / eps_decay)

    state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    
    steps_done += 1
    if sample > eps_treshold:
        with torch.no_grad():
            return model(state).max(1)[1].item()
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

replay_buffer = ReplayBuffer(20000)      

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.from_numpy(np.float32(state)).to(device)
    next_state = torch.from_numpy(np.float32(next_state)).to(device)    
    action     = torch.LongTensor(action).to(device)
    reward     = torch.FloatTensor(reward).to(device)
    done       = torch.FloatTensor(done).to(device)

    q_values      = model(state)
    next_q_values = model(next_state)    

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]    
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    
    loss = (q_value - expected_q_value.detach()).pow(2).mean()
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

def plot(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()

num_frames = 100000
batch_size = 32
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

start = time.time()

state = env.reset()
for frame_idx in range(1, num_frames + 1):    
    action = select_action(state)
    
    next_state, reward, done, _ = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    
    if frame_idx > num_frames-5000: env.render()

    if done:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0
        
    # if len(replay_buffer) > batch_size:
    if len(replay_buffer) % batch_size == 0:
        loss = compute_td_loss(batch_size)
        losses.append(loss.item())
        
    if frame_idx % 200 == 0:
        print(np.mean(all_rewards[-10:]))
        # plot(frame_idx, all_rewards, losses)

env.close()
print(f'It took {time.time()-start} secs using {device}')          