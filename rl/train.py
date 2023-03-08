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

import gym

curr_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(curr_path, ".."))
sys.path.append(os.path.join(curr_path, "..", "football2d_gym"))

import football2d
from rl.algorithms.a2c import A2C
from rl.envs import VectorEnvPyTorchWrapper, PyTorchRecordEpisodeStatistics

import ipdb

parser = argparse.ArgumentParser()

# algorithm
parser.add_argument("--algorithm", type=str, default="a2c",
                    choices=["a2c"],
                    help="training algorithm to choose")
# environment hyperparams
parser.add_argument("--n_envs", type=int, default=10,
                    help="number of envs")
parser.add_argument("--n_updates", type=int, default=1000,
                    help="number of updates")
parser.add_argument("--n_steps_per_update", type=int, default=128,
                    help="number of steps per update")
parser.add_argument("--randomize_domain", action="store_true",
                    help="randomize env params")
parser.add_argument("--lunarlander", action="store_true",
                    help="experiment with lunarlander")
parser.add_argument("--time_limit", type=int, default=20,
                    help="time limit of football2d")
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
# training setting
parser.add_argument("--use_cuda", action="store_true",
                    help="use GPU")
parser.add_argument("--seed", type=int, default=42,
                    help="random seed")
# save model
parser.add_argument("--name", type=str, default="default",
                    help="model name to save")

args = parser.parse_args()

writer_path = os.path.join("logs", args.algorithm, args.name)
writer = SummaryWriter(writer_path)

# set the device
if args.use_cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

if args.randomize_domain:
    envs = gym.vector.make("football2d/SelfTraining-v0", num_envs=args.n_envs, randomize_position=True,
                           time_limit=args.time_limit)
else:
    envs = gym.vector.make("football2d/SelfTraining-v0", num_envs=args.n_envs, randomize_position=False, 
                           time_limit=args.time_limit)

if args.lunarlander:
    if args.randomize_domain:
        envs = gym.vector.AsyncVectorEnv(
            [
                lambda: gym.make(
                    "LunarLanderContinuous-v2",
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
        envs = gym.vector.make("LunarLanderContinuous-v2", num_envs=args.n_envs, max_episode_steps=1000)

envs = VectorEnvPyTorchWrapper(envs, device)

if envs.single_observation_space.__class__.__name__ == "Dict":
    args.obs_shape = sum([obs_box.shape[0] for key, obs_box in envs.single_observation_space.items()])
else:
    args.obs_shape = envs.single_observation_space.shape[0]
args.action_space_type = envs.single_action_space.__class__.__name__
args.action_shape = envs.single_action_space.shape[0]
#if args.action_space_type == "Discrete":
#    args.action_shape = envs.single_action_space.n
#elif args.action_space_type == "Box":
#    args.action_shape = envs.single_action_space.shape[0]
#elif args.action_space_type == "MultiBinary":
#    args.action_shape = envs.single_action_space.shape[0]
#else:
#    raise NotImplementedError


# init the agent
agent = A2C(args.obs_shape, args.action_shape, args.action_space_type, 
            device, args.critic_lr, args.actor_lr, args.n_envs)

# create a wrapper environment to save episode returns and episode lengths
envs_wrapper = PyTorchRecordEpisodeStatistics(envs, deque_size=args.n_envs * args.n_updates, device=device)

episode_rewards = deque(maxlen=args.n_envs)
episode_lengths = deque(maxlen=args.n_envs)

# train the agent
for sample_phase in tqdm(range(args.n_updates)):

    # we don't have to reset the envs, they just continue playing
    # until the episode is over and then reset automatically

    # reset lists that collect experiences of an episode (sample phase)
    ep_value_preds = torch.zeros(args.n_steps_per_update, args.n_envs, device=device)
    ep_rewards = torch.zeros(args.n_steps_per_update, args.n_envs, device=device)
    ep_action_log_probs = torch.zeros(args.n_steps_per_update, args.n_envs, device=device)
    masks = torch.zeros(args.n_steps_per_update, args.n_envs, device=device)

    # at the start of training reset all envs to get an initial state
    if sample_phase == 0:
        states, info = envs_wrapper.reset(seed=args.seed)

    # play n steps in our parallel environments to collect data
    for step in range(args.n_steps_per_update):

        # select an action A_{t} using S_{t} as input for the agent
        actions, action_log_probs, state_value_preds, entropy = agent.select_action(
            states
        )

        # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
        states, rewards, terminated, truncated, infos = envs_wrapper.step(
            actions.detach()
        )

        if "_episode" in infos.keys() and "episode":
            if infos["_episode"].any():
                for ep_ind in np.flatnonzero(infos["_episode"]):
                    episode_rewards.append(infos['episode']['r'][ep_ind])
                    episode_lengths.append(infos['episode']['l'][ep_ind])

        ep_value_preds[step] = torch.squeeze(state_value_preds)
        ep_rewards[step] = rewards
        ep_action_log_probs[step] = action_log_probs

        # add a mask (for the return calculation later);
        # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)

        #masks[step] = torch.tensor([not term for term in terminated])
        masks[step] = torch.cat([term.unsqueeze(0) for term in terminated]).eq(0).float()

    # calculate the losses for actor and critic
    critic_loss, actor_loss = agent.get_losses(
        ep_rewards,
        ep_action_log_probs,
        ep_value_preds,
        entropy,
        masks,
        args.gamma,
        args.lam,
        args.ent_coef,
        device,
    )

    # update the actor and critic networks
    agent.update_parameters(critic_loss, actor_loss)

    # log the losses and entropy
    if len(episode_lengths) > 0:
        writer.add_scalar("training/episode_length", np.mean(episode_lengths), sample_phase)
    if len(episode_rewards) > 0:
        writer.add_scalar("training/episode_reward", np.mean(episode_rewards), sample_phase)
    writer.add_scalar("training/entropy", entropy.mean(), sample_phase)
    writer.add_scalar("training/actor_loss", actor_loss, sample_phase)
    writer.add_scalar("training/critic_loss", critic_loss, sample_phase)
    # observation
    writer.add_scalar("observation/normal_std", agent.dist.logstd.exp().mean(), sample_phase)




# save model
save_dir = os.path.join("saved_models", args.algorithm, args.name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

hyperparams_path = os.path.join(save_dir, "hyperparams.json")
actor_weights_path = os.path.join(save_dir, "actor_weights.pt")
critic_weights_path = os.path.join(save_dir, "critic_weights.pt")

json.dump(args.__dict__, open(hyperparams_path, "w"), indent=4)
torch.save(agent.actor.state_dict(), actor_weights_path)
torch.save(agent.critic.state_dict(), critic_weights_path)
print("Successfully save model in {}".format(save_dir))


# end training
envs.close()
writer.close()


