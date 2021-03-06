import math, random, time, os

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

# env = gym.envs.make("BreakoutNoFrameskip-v4")

from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from collections import deque

env_id = "BreakoutNoFrameskip-v4"
env    = make_atari(env_id)
env    = wrap_deepmind(env, frame_stack=True)
env    = wrap_pytorch(env)

# Replay Buffer
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

class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),            
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
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def act(self, state, epsilon):        
        if random.random() > epsilon:
            state = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)

            with torch.no_grad():
                q_value = self.forward(state)
            action  = q_value.max(1)[1].item()
        else:
            action = random.randrange(env.action_space.n)
        return action

def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())

def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.from_numpy(np.float32(state)).to(device)
    next_state = torch.from_numpy(np.float32(next_state)).to(device)    
    action     = torch.LongTensor(action).to(device)
    reward     = torch.FloatTensor(reward).to(device)
    done       = torch.FloatTensor(done).to(device)

    q_values      = current_model(state)
    next_q_values = target_model(next_state)    

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]    
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

current_model = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)
target_model = CnnDQN(env.observation_space.shape, env.action_space.n).to(device)

update_target(current_model, target_model)
   
optimizer = optim.Adam(current_model.parameters(), lr=0.0001)

replay_initial = 5000
replay_buffer = ReplayBuffer(200000)

epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 5000000

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

plt.plot([epsilon_by_frame(i) for i in range(200000)])
plt.show()

num_frames = 20000000
batch_size = 64
gamma      = 0.99

losses = []
all_rewards = []
episode_reward = 0

start = time.time()
replay_done = False
batch_pushed = 0

epoch = 1
load = True
if load:
    checkpoint = torch.load(f'checkpoints/dqn_breakout/checkpoint.pth')
    current_model.load_state_dict(checkpoint['current_state_dict'])
    target_model.load_state_dict(checkpoint['current_state_dict'])
    optimizer.load_state_dict(checkpoint['policy_optim_dict'])
    current_model.train()
    target_model.train()
    epoch = checkpoint['epoch']


state = env.reset()
for frame_idx in range(epoch+1, num_frames + 1):
    epsilon = epsilon_by_frame(frame_idx)
    action = current_model.act(state, epsilon)
    # action = select_action(state)
    
    next_state, reward, done, info = env.step(action)
    replay_buffer.push(state, action, reward, next_state, done)
    
    state = next_state
    episode_reward += reward
    batch_pushed += 1

    # debugging
    if frame_idx > num_frames-10000: env.render()
    # env.render()
    # time.sleep(0.04)
    
    if done and info['ale.lives'] == 0:
        state = env.reset()
        all_rewards.append(episode_reward)
        episode_reward = 0

    if len(replay_buffer) > replay_initial:
        if replay_done is False:
            replay_done = True
        if batch_pushed >= batch_size:
            loss = compute_td_loss(batch_size)
            losses.append(loss.item())
            batch_pushed = 0
        
    if frame_idx % 10000 == 0 and replay_done == True:
        print(f'Reward: {np.mean(all_rewards[-40:])}, step: {round((frame_idx/num_frames*100), 2)}%')
        # plot(frame_idx, all_rewards, losses)
        
        print(f'{round(time.time()-start)}secs')
        start = time.time()

    if frame_idx % 10000 == 0:
        update_target(current_model, target_model)

    # Save
    if frame_idx % 100000 == 0:
        # save the model
        checkpoint_path = 'checkpoints/dqn_breakout/'
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save({
            'epoch': frame_idx,
            'current_state_dict': current_model.state_dict(),
            'policy_optim_dict': optimizer.state_dict()}, checkpoint_path + f'checkpoint.pth')
        print(f'saved heeey')
