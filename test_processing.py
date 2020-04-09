import gym
import cv2
import numpy as np

from common.atari_wrappers import make_atari, wrap_deepmind

import torch
import torch.nn as nn
import torch.nn.functional as F

env_name = 'BreakoutNoFrameskip-v0'

env = make_atari(env_name)
env = wrap_deepmind(env)

state = env.reset()


import matplotlib.pyplot as plt
plt.ion()

# plt.imshow(np.array(state).squeeze(2))


# cv2.imshow('test', np.array(state))
# cv2.waitKey(0)



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



state = state.reshape(1, 84, 84)

model = CnnDQN(state.shape, 4)

state_tensor = torch.Tensor(state).unsqueeze(0)

print(model(state_tensor))



# for i in range(100):
#     next_state, reward, done, _ = env.step(1)
#     if i == 90:
#         print('heey')

# print(state)

# next_state, reward, done, _ = env.step(1)

# print(next_state)
