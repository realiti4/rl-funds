import gym
import time
from itertools import count
import random
import numpy as np
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

device = torch.device('cuda')

env = gym.make('gym_backgammon:backgammon-v0')

class value_func(nn.Module):
    def __init__(self):
        super(value_func, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

class Agent:
    def __init__(self, color):
        self.color = color
        self.name = f'AgentExample({self.color})'

    def roll_dice(self):
        return (-random.randint(1, 6), -random.randint(1, 6)) if self.color == WHITE else (random.randint(1, 6), random.randint(1, 6))

    def choose_best_action(self, actions):
        best_action = None

        if actions:           
            values = [0.0 for i in range(len(actions))]
            # values = [0.0] * len(actions)

            tmp_counter = env.counter
            env.counter = 0
            saved_state = env.game.save_state()        

            for i, action in enumerate(actions):            

                observation, reward, done, info = env.step(action)
                observation = torch.tensor(observation).unsqueeze(0).to(device)

                values[i] = model(observation).detach()

                env.game.restore_state(saved_state)

            best_action_index = torch.cat(values).max(0)[1] if self.color == 0 else torch.cat(values).min(0)[1]
            # best_action_index = int(np.argmax(values)) if self.color == 'WHITE' else int(np.argmax(values))
            best_action = list(actions)[best_action_index]

            # next_value = torch.cat(values).max() if self.color == 0 else torch.cat(values).min()
            return best_action
        # return random.choice(list(actions)) if actions else None

model = value_func().to(device)
optimizer = optim.Adam(model.parameters())

wins = {WHITE: 0, BLACK: 0}

# agents = {WHITE: Agent(WHITE), BLACK: Agent(BLACK)}

# agent_color, first_roll, observation = env.reset()

# agent = agents[agent_color]

# t = time.time()

env.render(mode='human')

gamma = 0.99

for i in range(1000):
    agents = {WHITE: Agent(WHITE), BLACK: Agent(BLACK)}
    agent_color, first_roll, observation = env.reset()
    agent = agents[agent_color]
    t = time.time()
    for i in count():
        if first_roll:
            roll = first_roll
            first_roll = None
        else:
            roll = agent.roll_dice()

        # print("Current player={} ({} - {}) | Roll={}".format(agent.color, TOKEN[agent.color], COLORS[agent.color], roll))

        # TODO action select
        state = torch.tensor(observation).unsqueeze(0).to(device)
        value = model(state)
        
        actions = env.get_valid_actions(roll)

        action = agent.choose_best_action(actions)

        observation_next, reward, done, winner = env.step(action)

        # env.render(mode='human')

        next_state = torch.tensor(observation).unsqueeze(0).to(device)
        next_value = model(next_state)

        expected_value = reward + gamma * next_value * (1 - done)
        loss = (value - expected_value.detach()).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if done:
            if winner is not None:
                wins[winner] += 1

            tot = wins[WHITE] + wins[BLACK]
            tot = tot if tot > 0 else 1

            print("Game={} | Winner={} after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(1, winner, i,
                agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))

            break

        agent_color = env.get_opponent_agent()
        agent = agents[agent_color]
        observation = observation_next

env.close()

