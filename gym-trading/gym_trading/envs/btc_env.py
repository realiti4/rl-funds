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

        # rewards test
        self.liq = 1000
        self.btc = 0
        self.current_price = None

        self.current_tick = 0

        print('test')

    def step(self, action):
        """
        sell = 0
        buy = 1
        wait = 2

        action = (sell/buy, amount)
        """
        fixed_amount = 100

        position, amount_size = action
        amount = fixed_amount * amount_size
        trading_fee = (amount / 100) * 0.075

        if position == 0:
            self.liq += amount - trading_fee
            self.btc -= amount / self.current_price
        elif position == 1:
            self.liq -= amount + trading_fee
            self.btc += amount / self.current_price
        
        self.current_tick += 1

        next_state = self.get_observation()        

        if self.current_tick == self.state_window - self.history_size:
            done = True
            reward = self.calculate_rewards()

            self.reset()
        else:
            done = False
            reward = 0

        info = None

        return next_state, reward, done, info


    def reset(self):
        self.liq = 1000
        self.btc = 0
        self.current_price = None
        self.current_tick = 0

        key = np.random.randint(self.len-self.state_window)

        self.array_window = self.array[key:key+self.state_window]
        state = self.get_observation()

        # self.current_price = state[-1][0]

        print('debugging')

        return state

    def get_observation(self):
        state = self.array_window[self.current_tick:self.current_tick+self.history_size]
        self.current_price = state[-1][0]
        return state

    def calculate_rewards(self):
        total_liq = (self.btc * self.current_price) + self.liq
        return total_liq - 1000     # TODO set fixed starting liq


    def render(self, mode='human', close=False):
        print('heeey')





# print(df.tail())