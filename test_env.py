import gym
import gym_trading
import pandas as pd

from gym_trading.envs.pre_new import preprocessing_v3

import matplotlib.pyplot as plt

env = gym.make('btc-v0')

df = pd.read_csv('gym-trading/gym_trading/envs/data/bitfinex_1h_downloaded.csv')

print(df.tail())


preprocess_train = preprocessing_v3(df, ['close', 'open', 'low', 'high', 'volume'])
# array_train, _ = preprocess_train.preprocess_data(360, 48, 
#                                                 percentage_change=True, center_zero=False, smoothing=1, 
#                                                 convert_to_log=1)

array = preprocess_train.temp_handler(convert_to_log=0, normalize=False)

# plt.plot(array[:,0])
# plt.show()

state = env.reset()

print('debugging')

