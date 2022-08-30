import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

from environment import Env
from parameters import Parameters

pa = Parameters()
pa.compute_dependent_parameters()

env = Env(pa, render=False, repre='compact')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

env = Env(pa, render=True, repre='compact')
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    print(action)
    # action = env.action_space.sample()
    obs, rewards, done, info = env.step(action)
    if done:
        print('done')
        break



# check_env(env=env)