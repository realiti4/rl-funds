import math
import random, os

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


from stable_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind

env_name = 'BreakoutNoFrameskip-v4'

def make_test():
    env = make_atari(env_name)
    env = wrap_deepmind(env, frame_stack=True)
    return env

# envs = [make_test for env in range(2)]

# envs = SubprocVecEnv(envs)


if __name__ == '__main__':
    envs = [make_test for env in range(2)]
    envs = SubprocVecEnv(envs)

    state = envs.reset()

    print('success!!!')