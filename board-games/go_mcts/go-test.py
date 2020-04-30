import gym
import numpy as np

env = gym.make('gym_go:go-v0', size=5, reward_method='real')

first_action = (2,4)
second_action = (4,2)
state, reward, done, info = env.step(first_action)
env.render('terminal')

state, reward, done, info = env.step(second_action)
env.render('terminal')




class Actions():
    def __init__(self):
        self.valid_actions = np.ones([3, 3], dtype='uint8')

    def select_action(self, i):
        indexes = np.nonzero(self.valid_actions)
        if not indexes[0].size == 0 or not indexes[0].size == 0:
            x = np.random.choice(np.unique(indexes[0]), 1)
            y = np.random.choice(np.unique(indexes[1]), 1)
            
            if self.valid_actions[x, y] == 0:
                print('hee')
            assert self.valid_actions[x, y] != 0
            self.valid_actions[x, y] = 0
        else:
            print('heey')
            return None
        return (x, y)


class Model():
    def __init__(self):
        pass

    def forward(self, x):
        pass


def train_agent():
    pass


# a = Actions()


# for i in range(100):
#     action = a.select_action(i)


print('debugging')