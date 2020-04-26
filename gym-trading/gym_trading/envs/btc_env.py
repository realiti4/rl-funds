import pandas as pd

import gym
from gym import error, spaces, utils
from gym.utils import seeding

from gym_trading.envs.pre_processing import preprocessing_v3


class btc_env(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        ...
    def step(self, action):
        ...
    def reset(self):
        ...
    def render(self, mode='human', close=False):
        print('heeey')

import os

this_is_awesome = os.path.dirname(__file__)

df = pd.read_csv(this_is_awesome + '/data/bitfinex_1h_downloaded.csv')



# print(df.tail())