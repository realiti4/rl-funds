import gym

env = gym.make('gym_go:go-v0', size=7, reward_method='real')

first_action = (2,5)
second_action = (5,2)
state, reward, done, info = env.step(first_action)
env.render('terminal')

state, reward, done, info = env.step(second_action)
env.render('terminal')


