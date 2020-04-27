import numpy as np
import pandas as pd

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_trading.envs.pre_processing import preprocessing_v3

import os

class btc_env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        path = os.path.dirname(__file__)
        columns = ['close', 'open', 'low', 'high']

        self.df = pd.read_csv(path + '/data/bitfinex_1h_downloaded.csv')        
        self.len = len(self.df)

        self.array = self.df[columns].to_numpy()

        self.state_window = 5000
        self.history_size = 360

        print('test')

    def step(self, action):
        pass

    def reset(self):
        key = np.random.randint(self.len-self.state_window)

        self.array_window = self.array[key:key+self.state_window]
        state = self.array_window[:self.history_size]

        print('debugging')

        return state

    def get_observation(self):
        pass

    def calculate_rewards(self):
        pass


    def render(self, mode='human', close=False):
        print('heeey')





# print(df.tail())