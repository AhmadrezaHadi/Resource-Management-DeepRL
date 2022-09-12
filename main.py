from statistics import mode
from typing import Callable
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN

from environment import Env
from parameters import Parameters
from utils import calculate_average_slowdown, SJF


if __name__ == '__main__':
    pa = Parameters()
    pa.compute_dependent_parameters()

    env = Env(pa, render=False, repre='image', end='all_done')
    obs = env.reset()

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
    iters = 1
    TIMESTEPS = 20_000
    while True:
        model.learn(total_timesteps=TIMESTEPS, tb_log_name="num_nw=10 Image sim_len=200 num_ex=1 job_rate=0.7", reset_num_timesteps=False)
        model.save(f"tmp/ppo_{iters}")
        iters += 1
        if iters > 50:
            break

    # model.load("tmp/ppo_11")
    # while True:
    #     action, _states = model.predict(obs)
    #     # action = SJF(env)
    #     # print(action)
    #     # action = env.action_space.sample()
    #     obs, rewards, done, info = env.step(action)
    #     if done:
    #         print('done')
    #         print(f"average slowdown = {calculate_average_slowdown(info=info)}")
    #         break



# check_env(env=env)