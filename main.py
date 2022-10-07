from ast import parse
from statistics import mode
from subprocess import call
from typing import Callable
import gym
import numpy as np
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import DQN
import argparse
from environment import Env
from parameters import Parameters
from utils import calculate_average_slowdown, SJF

#####################################################################################
##########################            Arguments            ##########################
#####################################################################################
parser = argparse.ArgumentParser(description="")
parser.add_argument("mode", choices=['train', 'eval'],
                    help="Train or Eval model")
parser.add_argument("algorithm", choices=['ppo', 'dqn'],
                    help="algorithm for training")
parser.add_argument("-t", "--timesteps", default=500_000, type=int,
                    help="Timesteps for training")
parser.add_argument("-r", "--render", default=False, type=bool,
                    help="Render the output of environment or not")
parser.add_argument("-re", "--representation", default='compact', type=str,
                    help='state returned from the environment (compact or image).')
parser.add_argument("-l", "--load", type=str, help="model path to load")

args = parser.parse_args()

RENDER = args.render
TIMESTEPS = args.timesteps
REPRE = args.representation


def eval_model(model, env):
    methods = ['Random', 'SJF', 'Model Algorithm']
    obs = env.reset()
    for m in methods:
        while True:
            if m == 'Random':
                action = env.action_space.sample()
            elif m == 'SJF':
                break
            else:
                action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                print('done')
                print(
                    f"{m} method average slowdown = {calculate_average_slowdown(info=info)}")
                break


if __name__ == '__main__':
    pa = Parameters()
    pa.compute_dependent_parameters()
    pa.unseen = True
    env = Env(pa, render=RENDER, repre=REPRE, end='all_done')
    env.reset()

    net = [128, 128, 128, 128, 128]
    policy_kwargs = {
        "net_arch": net
    }

    if args.mode == 'train':
        checkpoint_callback = CheckpointCallback(save_freq=20000,
                                                 save_path=f'./models/{args.algorithm}_3',
                                                 name_prefix=f'{args.algorithm}')

        if args.algorithm == 'ppo':
            print('creating model')
            model = PPO('MlpPolicy', env,
                        tensorboard_log='./tensorboard/', device='auto', policy_kwargs=policy_kwargs)
            try:
                print("training")
                model.learn(TIMESTEPS, callback=checkpoint_callback,
                            tb_log_name=f'{args.algorithm}_128_4layer')
            except:
                # model.save_replay_buffer('tmp/last_model.pkl')
                model.save('tmp/last_model')
                print(f"model trained using {args.algorithm} algorithm")
        elif args.algorithm == 'dqn':
            pass

    if args.mode == 'eval':
        if args.algorithm == 'ppo':
            if not args.load:
                raise "model path not specified (--load model_path)"
            model = PPO('MlpPolicy', env).load(args.load, env)
            eval_model(model, env)
        elif args.algorithm == 'dqn':
            pass
