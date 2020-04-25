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

device = torch.device('cpu')

torch.set_default_tensor_type('torch.DoubleTensor')

env = gym.make('gym_backgammon:backgammon-v0')

seed = 548

# torch.set_num_threads(3)

# torch.manual_seed(seed)
# random.seed(seed)

class value_func(nn.Module):
    def __init__(self, hidden_size):
        super(value_func, self).__init__()

        self.lr = 0.1
        self.lamda = 0.7
        self.eligibility_traces = None
        
        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], hidden_size),
            # nn.LayerNorm(40),
            nn.Sigmoid(),
            # nn.ReLU(),
            # nn.Linear(40128),
            # nn.ReLU(),

            nn.Linear(hidden_size, 1),
            # nn.Softmax(dim=1)
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = torch.tensor(x).to(device)
        return self.layers(x)

    def init_eligibility_traces(self):
        self.eligibility_traces = [torch.zeros(weights.shape, requires_grad=False) for weights in list(self.parameters())]

    def update_weights(self, p, p_next):
        # reset the gradients
        self.zero_grad()

        # compute the derivative of p w.r.t. the parameters
        p.backward()

        with torch.no_grad():

            td_error = p_next - p

            # get the parameters of the model
            parameters = list(self.parameters())

            for i, weights in enumerate(parameters):

                # z <- gamma * lambda * z + (grad w w.r.t P_t)
                self.eligibility_traces[i] = self.lamda * self.eligibility_traces[i] + weights.grad

                # w <- w + alpha * td_error * z
                new_weights = weights + self.lr * td_error * self.eligibility_traces[i]
                weights.copy_(new_weights)

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
            assert_color = self.color

            tmp_counter = env.counter
            env.counter = 0
            saved_state = env.game.save_state()        

            for i, action in enumerate(actions):            

                observation, reward, done, info = env.step(action)                
                
                # 2-ply
                self.color = env.get_opponent_agent()   
                roll = self.roll_dice()
                actions_2ply = env.get_valid_actions(roll)

                if actions_2ply:                   
                    values_temp = [0.0 for i in range(len(actions_2ply))]

                    tmp_counter_2ply = env.counter
                    saved_state_2ply = env.game.save_state()

                    for i_e, action_2ply in enumerate(actions_2ply):
                        observation_2ply, reward, done, info = env.step(action_2ply)

                        with torch.no_grad():
                            values_temp[i_e] = model(observation_2ply)  

                        env.game.restore_state(saved_state_2ply)

                    # env.game.restore_state(saved_state)
                    self.color = env.get_opponent_agent()
                    assert assert_color == self.color
                    values[i] = torch.cat(values_temp).min().unsqueeze(0) if self.color == 0 else torch.cat(values_temp).max().unsqueeze(0)
                else:
                    self.color = env.get_opponent_agent()
                    assert assert_color == self.color
                    with torch.no_grad():
                        values[i] = model(observation)      # detach() ???

                env.game.restore_state(saved_state)

            assert assert_color == self.color
            best_action_index = torch.cat(values).max(0)[1] if self.color == 0 else torch.cat(values).min(0)[1]
            # best_action_index = int(np.argmax(values)) if self.color == 'WHITE' else int(np.argmax(values))
            best_action = list(actions)[best_action_index]
            env.counter = tmp_counter

            return best_action

def checkpoint(checkpoint_path, step):
    path = checkpoint_path + f"/test_2ply_new.tar"
    torch.save({'step': step + 1, 'model_state_dict': model.state_dict(), 'eligibility': model.eligibility_traces if model.eligibility_traces else []}, path)
    print("\nCheckpoint saved: {}".format(path))


model = value_func(hidden_size=80).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Params
start_episode = 1
num_episodes = 100000
eligibility = True
gamma = 0.99

load = True


if load:
    cp_state = torch.load('board-games/td_gammon/saved/test.tar')
    model.load_state_dict(cp_state['model_state_dict'])
    model.eligibility_traces = cp_state['eligibility']
    start_episode = cp_state['step']
    model.train()       # check this not sure    


def train_agent():
    # start_episode = start_episode
    global num_episodes
    num_episodes += start_episode

    wins = {WHITE: 0, BLACK: 0}

    agents = {WHITE: Agent(WHITE), BLACK: Agent(BLACK)}

    durations = []
    steps = 0

    for i_episode in range(start_episode, num_episodes):
        
        if eligibility:
            model.init_eligibility_traces()

        agent_color, first_roll, observation = env.reset()
        agent = agents[agent_color]

        t = time.time()

        for i in count():
            if first_roll:
                roll = first_roll
                first_roll = None
            else:
                roll = agent.roll_dice()

            # observation = torch.tensor(observation).to(device)
            value = model(observation)
            
            actions = env.get_valid_actions(roll)
            action = agent.choose_best_action(actions)

            observation_next, reward, done, winner = env.step(action)

            # observation_next = torch.tensor(observation_next).to(device)
            next_value = model(observation_next)

            # expected_value = reward + gamma * next_value * (1 - done)
            # loss = (value - expected_value.detach()).pow(2).mean()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            if done:
                if winner is not None:
                    loss = model.update_weights(value, reward)

                    wins[agent.color] += 1

                tot = sum(wins.values())
                tot = tot if tot > 0 else 1

                print("Game={:<6d} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(i_episode + 1, winner, i,
                    agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                    agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))

                durations.append(time.time() - t)
                steps += i
                break
            else:
                loss = model.update_weights(value, next_value)

            agent_color = env.get_opponent_agent()
            agent = agents[agent_color]

            observation = observation_next

        # Save
        if i_episode % 1000 == 0:
            checkpoint(checkpoint_path='board-games/td_gammon/saved', step=i_episode)
            

train_agent()

env.close()