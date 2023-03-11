import os
import sys
import argparse
import json
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pprint import pprint

import gym

from pathlib import Path
parent_dir = Path(__file__).absolute().parent.parent
sys.path.append(os.path.abspath(parent_dir))
sys.path.append(os.path.join(parent_dir, "football2d_gym"))

import football2d
from rl.algorithms.a2c import A2C
from rl.algorithms.ppo import PPO
from rl.envs import VectorEnvPyTorchWrapper, PyTorchRecordEpisodeStatistics
from rl.storage import TrainingMemory
from rl.rewards import get_auxiliary_reward_manager

import ipdb

parser = argparse.ArgumentParser()

#############################################################################
# Recommend for a2c
#############################################################################
# --n_updates 30000
# --n_steps_per_update 8
# --hidden_size 128
# --actor_lr 1e-3
# --critic_lr 5e-3
# --save_interval 1000
# --ent_coef 0.001
#############################################################################


# algorithm
parser.add_argument("--algorithm", type=str, default="ppo",
                    choices=["a2c", "ppo"],
                    help="training algorithm to choose")
# environment
parser.add_argument("--env_name", type=str, default="SelfTraining-v0",
                    choices=["SelfTraining-v0", "SelfTraining-v1", "SelfTraining-v2"],
                    help="Football2d environment to choose")
parser.add_argument("--lunarlander", action="store_true",
                    help="experiment with lunarlander")
# environment hyperparams
parser.add_argument("--n_envs", type=int, default=16,
                    help="number of envs")
parser.add_argument("--n_updates", type=int, default=500,
                    help="number of updates")
parser.add_argument("--n_steps_per_update", type=int, default=128,
                    help="number of steps per update, recommend small for a2c and big for ppo")
parser.add_argument("--randomize_domain", action="store_true",
                    help="randomize env params")
parser.add_argument("--time_limit", type=int, default=20,
                    help="time limit of football2d")
# actor-critic model
parser.add_argument("--hidden_size", type=int, default=128,
                    help="hidden size of actor and critic")
parser.add_argument("--init_sample_scale", type=float, default=1.0,
                    help="initial scale for normal sampling")
parser.add_argument("--normalize_factor", type=float, default=1e-3,
                    help="normalize state vector to be appropriate")
parser.add_argument("--output_activation", type=str, default="none",
                    help="output activation function of the actor")
# agent hyperparams
parser.add_argument("--gamma", type=float, default=0.999,
                    help="discount factor for reward")
parser.add_argument("--lam", type=float, default=0.95,
                    help="GAE hyperparameter. lam=1 corresponds to MC sampling; lam=0 corresponds to TD-learning.")
parser.add_argument("--ent_coef", type=float, default=0.001,
                    help="coefficient for the entropy bonus (to encourage exploration)")
parser.add_argument("--actor_lr", type=float, default=1e-3,
                    help="actor learning rate")
parser.add_argument("--critic_lr", type=float, default=5e-3,
                    help="critic learning rate")
parser.add_argument("--max_grad_norm", type=float, default=0.5,
                    help="clip gradient, max norm of gradient")
# ppo
parser.add_argument("--batch_size", type=int, default=64,
                    help="batch size in ppo")
parser.add_argument("--n_ppo_epochs", type=int, default=5,
                    help="number of epochs to train in ppo")
parser.add_argument("--clip_param", type=float, default=0.2,
                    help="ppo clip parameter")
# training setting
parser.add_argument("--use_cuda", action="store_true",
                    help="use GPU")
parser.add_argument("--seed", type=int, default=42,
                    help="random seed")
parser.add_argument("--use_auxiliary_reward", action="store_true",
                    help="use auxiliary reward")
parser.add_argument("--auxiliary_reward_type", type=str, default="default",
                    help="auxiliary reward type to choose")
# save model
parser.add_argument("--name", type=str, default="default",
                    help="model name to save")
parser.add_argument("--save_interval", type=int, default=50,
                    help="interval steps for saving models")

args = parser.parse_args()

# set the device
if args.use_cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# set the environment
full_env_name = "football2d/{}".format(args.env_name)
if args.randomize_domain:
    envs = gym.vector.make(full_env_name, num_envs=args.n_envs, randomize_position=True,
                           time_limit=args.time_limit)
else:
    envs = gym.vector.make(full_env_name, num_envs=args.n_envs, randomize_position=False, 
                           #render_mode="human",
                           time_limit=args.time_limit)

if args.lunarlander:
    args.env_name = "LunarLanderContinuous-v2"
    args.normalize_factor = 1.0
    if args.randomize_domain:
        envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make(
                    args.env_name,
                    gravity=np.clip(
                        np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01
                    ),
                    enable_wind=np.random.choice([True, False]),
                    wind_power=np.clip(
                        np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99
                    ),
                    turbulence_power=np.clip(
                        np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99
                    ),
                    max_episode_steps=1000,
                )
                for i in range(args.n_envs)
            ]
        )
    
    else:
        envs = gym.vector.make(args.env_name, num_envs=args.n_envs, max_episode_steps=1000)

envs = VectorEnvPyTorchWrapper(envs, device)

if envs.single_observation_space.__class__.__name__ == "Dict":
    args.obs_shape = sum([obs_box.shape[0] for key, obs_box in envs.single_observation_space.items()])
else:
    args.obs_shape = envs.single_observation_space.shape[0]
args.action_shape = envs.single_action_space.shape[0]


# save model dir
save_dir = os.path.join("saved_models", args.env_name, args.algorithm, args.name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def save_model(dir_name):
    hyperparams_path = os.path.join(dir_name, "hyperparams.json")
    actor_weights_path = os.path.join(dir_name, "actor_weights.pt")
    critic_weights_path = os.path.join(dir_name, "critic_weights.pt")
    
    json.dump(args.__dict__, open(hyperparams_path, "w"), indent=4)
    torch.save(agent.actor.state_dict(), actor_weights_path)
    torch.save(agent.critic.state_dict(), critic_weights_path)
    print("Successfully save model in {}".format(dir_name))

# init SummaryWriter
writer_path = os.path.join("logs", args.env_name, args.algorithm, args.name)
writer = SummaryWriter(writer_path)

# add Text for hyperparams
def pretty_json(hp):
    json_hp = json.dumps(hp, indent=4)
    return "".join("\t" + line for line in json_hp.splitlines(True))
writer.add_text("hyperparams", pretty_json(args.__dict__))

# init the agent
if args.algorithm == "a2c":
    agent = A2C(args.obs_shape, 
                args.action_shape, 
                args.hidden_size, 
                args.output_activation,
                device, 
                args.critic_lr, 
                args.actor_lr, 
                args.init_sample_scale,
                args.n_envs,
                args.ent_coef,
                args.max_grad_norm,
                normalize_factor=args.normalize_factor
                )
elif args.algorithm == "ppo":
    agent = PPO(args.obs_shape, 
                args.action_shape, 
                args.hidden_size, 
                args.output_activation,
                device, 
                args.batch_size,
                args.n_ppo_epochs,
                args.clip_param,
                args.critic_lr, 
                args.actor_lr, 
                args.init_sample_scale,
                args.n_envs,
                args.ent_coef,
                args.max_grad_norm,
                normalize_factor=args.normalize_factor
                )
else:
    raise KeyError("algorithm {} not implemented".format(args.algorithm))


if args.use_auxiliary_reward:
    auxiliary_reward_manager = get_auxiliary_reward_manager(args.env_name, device, args.auxiliary_reward_type)
else:
    auxiliary_reward_manager = None
# create a wrapper environment to save episode returns and episode lengths
envs_wrapper = PyTorchRecordEpisodeStatistics(envs, deque_size=args.n_envs * args.n_updates, device=device,
                                              auxiliary_reward_manager=auxiliary_reward_manager)


episode_rewards = deque(maxlen=args.n_envs)
episode_lengths = deque(maxlen=args.n_envs)
episode_final_observations = deque(maxlen=args.n_envs)
episode_final_infos = deque(maxlen=args.n_envs)

memory = TrainingMemory(args.n_steps_per_update, args.n_envs, args.obs_shape, args.action_shape, device)

states, info = envs_wrapper.reset(seed=args.seed)
memory.states.append(states)


# train the agent
for update_step in tqdm(range(args.n_updates)):

    # we don't have to reset the envs, they just continue playing
    # until the episode is over and then reset automatically
    #
    # at the start of training reset all envs to get an initial state

    # play n steps in our parallel environments to collect data
    for step in range(args.n_steps_per_update):

        # select an action A_{t} using S_{t} as input for the agent
        with torch.no_grad():
            actions, action_log_probs, state_value_preds, entropy = agent.select_action(
                states
            )
        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        states, rewards, terminated, truncated, infos = envs_wrapper.step(
            actions
        )

        if "_episode" in infos.keys() and "episode":
            if infos["_episode"].any():
                for ep_ind in np.flatnonzero(infos["_episode"]):
                    episode_rewards.append(infos['episode']['r'][ep_ind])
                    episode_lengths.append(infos['episode']['l'][ep_ind])
                    episode_final_observations.append(infos["final_observation"][ep_ind])
                    episode_final_infos.append(infos["final_info"][ep_ind])

        # add a mask (for the return calculation later);
        # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)

        #masks[step] = torch.tensor([not term for term in terminated])
        masks = torch.cat([term.unsqueeze(0) for term in terminated]).eq(0).float().to(device)

        memory.insert(states, actions, action_log_probs, state_value_preds, rewards, masks)

    # get next value V(s_{t+1})
    with torch.no_grad():
        next_state_value, _ = agent.forward(states)
        next_state_value = next_state_value.squeeze().detach()

    # calculate the losses for actor and critic
    memory.compute_returns_and_advantages(args.gamma, args.lam, next_state_value)

    # update the actor and critic networks
    critic_loss, actor_loss, entropy = agent.update_parameters(memory)

    # clear memory
    memory.after_update()

    # log the losses and entropy
    if len(episode_lengths) > 0:
        writer.add_scalar("training/episode_length", np.mean(episode_lengths), update_step)
    if len(episode_rewards) > 0:
        writer.add_scalar("training/episode_reward", np.mean(episode_rewards), update_step)
    writer.add_scalar("training/entropy", entropy.mean(), update_step)
    writer.add_scalar("training/actor_loss", actor_loss, update_step)
    writer.add_scalar("training/critic_loss", critic_loss, update_step)
    # observation
    writer.add_scalar("observation/normal_std", agent.dist.logstd.exp().mean() * agent.dist.init_scale, update_step)
    # final observation
    if len(episode_final_observations) > 0:
        if episode_final_observations[0].__class__.__name__ in ["dict"]:
            for obs_key in episode_final_observations[0].keys():
                obs_value = np.concatenate([obs[obs_key][None, :] for obs in episode_final_observations])
                obs_value_mean = np.mean(obs_value, axis=0)
                writer.add_scalar("{}_final_state/{}.x".format(args.env_name, obs_key), obs_value_mean[0], 
                                  update_step)
                writer.add_scalar("{}_final_state/{}.y".format(args.env_name, obs_key), obs_value_mean[1], 
                                  update_step)
        else:
            obs_value = np.concatenate([obs[None, :] for obs in episode_final_observations])
            obs_value_mean = np.mean(obs_value, axis=0)
            for obs_ind, obs_val in enumerate(obs_value_mean):
                writer.add_scalar("{}_final_state/state.{}".format(args.env_name, obs_ind), obs_val, update_step)
    # final info
    if len(episode_final_infos) > 0:
        for info_key in episode_final_infos[0].keys():
            info_value = [info[info_key] for info in episode_final_infos]
            info_value_mean = np.mean(info_value)
            writer.add_scalar("{}_final_info/{}".format(args.env_name, info_key), info_value_mean, update_step)

    if update_step > 0 and update_step % args.save_interval == 0:
        ckp_dir_name = "ckp_{}".format(update_step)
        ckp_dir = os.path.join(save_dir, ckp_dir_name)
        if not os.path.exists(ckp_dir):
            os.mkdir(ckp_dir)
        save_model(ckp_dir)




save_model(save_dir)

# end training
envs.close()
writer.close()

