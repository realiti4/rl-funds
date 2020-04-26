from gym.envs.registration import register

register(
    id='btc-v0',
    entry_point='gym_trading.envs:btc_env',
)