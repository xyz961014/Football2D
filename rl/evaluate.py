import sys
import os
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

import gym

curr_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(curr_path, ".."))
sys.path.append(os.path.join(curr_path, "..", "football2d_gym"))

import football2d
from rl.algorithms.a2c import A2C
from rl.envs import EnvPyTorchWrapper


import ipdb

parser = argparse.ArgumentParser()

# algorithm
parser.add_argument("--algorithm", type=str, default="a2c",
                    choices=["a2c"],
                    help="training algorithm to choose")
# env
parser.add_argument("--env_name", type=str, default="default",
                    choices=["default",
                             "SelfTraining-v0", "SelfTraining-v1", "SelfTraining-v2"],
                    help="Football2d environment to choose")
parser.add_argument("--time_limit", type=int, default=10,
                    help="time limit of football2d")
# params
parser.add_argument("--n_episodes", type=int, default=10,
                    help="number of episodes")
parser.add_argument("--randomize_domain", action="store_true",
                    help="randomize env params")
parser.add_argument("--silent", action="store_true",
                    help="do not run it in human render mode")
parser.add_argument("--seed", type=int, default=42,
                    help="random seed")
# load model
parser.add_argument("--load_dir", type=str, default="saved_models/LunarLanderContinuous-v2/a2c/default",
                    help="directory to load model")

args = parser.parse_args()


# use CPU to evaluate
device = torch.device("cpu")

""" load network weights """
hyperparams_path = os.path.join(args.load_dir, "hyperparams.json")
actor_weights_path = os.path.join(args.load_dir, "actor_weights.pt")
critic_weights_path = os.path.join(args.load_dir, "critic_weights.pt")

model_args = json.load(open(hyperparams_path, "r"))
agent = A2C(model_args["obs_shape"], 
            model_args["action_shape"], 
            model_args["hidden_size"], 
            device, 
            model_args["critic_lr"], 
            model_args["actor_lr"],
            model_args["init_sample_scale"],
            n_envs=1)

if model_args["lunarlander"]:
    full_env_name = model_args["env_name"]
else:
    if args.env_name == "default":
        full_env_name = "football2d/{}".format(model_args["env_name"])
    else:
        full_env_name = "football2d/{}".format(args.env_name)

if args.silent:
    render_mode = None
else:
    render_mode = "human"

agent.actor.load_state_dict(torch.load(actor_weights_path))
agent.critic.load_state_dict(torch.load(critic_weights_path))
agent.eval()


episode_rewards = []
episode_lengths = []
for episode in range(args.n_episodes):

    # create a new sample environment to get new random parameters
    if not model_args["lunarlander"]:
        if args.randomize_domain:
            env = gym.make(full_env_name, render_mode=render_mode, randomize_position=True, 
                           time_limit=args.time_limit)
        else:
            env = gym.make(full_env_name, render_mode=render_mode, randomize_position=False, 
                           time_limit=args.time_limit)
    else:
        if args.randomize_domain:
            env = gym.make(
                full_env_name,
                render_mode=render_mode,
                gravity=np.clip(
                    np.random.normal(loc=-10.0, scale=2.0), a_min=-11.99, a_max=-0.01
                ),
                enable_wind=np.random.choice([True, False]),
                wind_power=np.clip(
                    np.random.normal(loc=15.0, scale=2.0), a_min=0.01, a_max=19.99
                ),
                turbulence_power=np.clip(
                    np.random.normal(loc=1.5, scale=1.0), a_min=0.01, a_max=1.99
                ),
                max_episode_steps=2000,
            )
        else:
            env = gym.make(full_env_name, render_mode=render_mode, max_episode_steps=2000)

    env = EnvPyTorchWrapper(env, device)

    # get an initial state
    state, info = env.reset()

    # play one episode
    done = False
    episode_reward = 0
    episode_length = 0
    while not done:

        # select an action A_{t} using S_{t} as input for the agent
        with torch.no_grad():
            action, _, _, _ = agent.select_action(state)

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        state, reward, terminated, truncated, info = env.step(action.squeeze())
        episode_reward += reward
        episode_length += 1

        # update if the environment is done
        done = terminated or truncated

    print("Episode {:3} reward: {:5.3f} | Episode length: {:5}".format(episode + 1, episode_reward, episode_length))
    episode_rewards.append(episode_reward)
    episode_lengths.append(episode_length)

print("Average episode reward: {:5.3f} +- {:5.3f} | Average episode length: {:5.3f} +- {:5.3f}".format(
        np.mean(episode_rewards), np.std(episode_rewards),
        np.mean(episode_lengths), np.std(episode_lengths))
     )

env.close()


