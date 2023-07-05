import os
import sys
import re
import numpy as np
from copy import copy
import torch
import torch.nn as nn
from torch import optim
from collections import OrderedDict

import ipdb


from pathlib import Path
parent_dir = Path(__file__).absolute().parent.parent.parent
sys.path.append(os.path.abspath(parent_dir))

from rl.utils import ScaleParameterizedNormal, FourierEncoding

class ActorCritic(nn.Module):
    def __init__(self, model_name, feature_shape, action_shape, hidden_size, output_activation, 
                       device, init_scale, n_envs, normalize_factor=1.0, dropout=0.0, 
                       encoding_type="none", encoding_size=128, 
                       use_goal_state=False, goal_position=(550.0, 0.0)):
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.normalize_factor = normalize_factor
        self.model_name = model_name

        self.use_goal_state = use_goal_state
        self.goal_position = goal_position

        self.determinstic = False

        n_features = sum(feature_shape) if feature_shape.__class__.__name__ in ["list", "tuple"] else feature_shape
        n_actions = sum(action_shape) if action_shape.__class__.__name__ in ["list", "tuple"] else action_shape

        self.dist = ScaleParameterizedNormal(n_actions=n_actions, init_scale=init_scale).to(self.device)

        if model_name == "basic":
            critic_layers = [
                nn.Linear(n_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, 1),  # estimate V(s)
            ]

            actor_layers = [
                nn.Linear(n_features, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, n_actions),  
            ]

            if output_activation == "none":
                pass
            elif output_activation == "tanh":
                actor_layers.append(nn.Tanh())
            else:
                raise KeyError("Unknown output activation")

            # define actor and critic networks
            self.critic = nn.Sequential(*critic_layers)
            self.actor = nn.Sequential(*actor_layers)

        elif model_name == "basic_wo_dropout":
            critic_layers = [
                nn.Linear(n_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),  # estimate V(s)
            ]

            actor_layers = [
                nn.Linear(n_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions),  
            ]

            if output_activation == "none":
                pass
            elif output_activation == "tanh":
                actor_layers.append(nn.Tanh())
            else:
                raise KeyError("Unknown output activation")

            # define actor and critic networks
            self.critic = nn.Sequential(*critic_layers)
            self.actor = nn.Sequential(*actor_layers)

        elif model_name == "world":
            self.world_encoder = WorldEncoder(feature_shape, hidden_size, 
                                              dropout=dropout,
                                              encoding_type=encoding_type,
                                              encoding_size=encoding_size)
            self.critic = nn.Sequential(
                                nn.Linear(hidden_size, hidden_size),
                                nn.GELU(),
                                nn.Dropout(p=dropout),
                                nn.Linear(hidden_size, 1)
                            )
            self.actor = SubdividedActor(action_shape, hidden_size, output_activation, dropout)

        elif model_name == "world_gpt":
            self.world_encoder = WorldTransformerEncoder(feature_shape, hidden_size, 
                                                         dropout=dropout)
            self.critic = nn.Sequential(
                                nn.Linear(hidden_size, hidden_size),
                                nn.GELU(),
                                nn.Dropout(p=dropout),
                                nn.Linear(hidden_size, 1)
                            )
            self.actor = SubdividedActor(action_shape, hidden_size, output_activation, dropout)

        else:
            raise KeyError("Unknown model type")

        self.to(device)


    def eval(self):
        super().eval()
        if hasattr(self, "world_encoder"):
            self.world_encoder.eval()
        self.actor.eval()
        self.critic.eval()
        self.determinstic = True

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "world_encoder"):
            self.world_encoder.train(mode)
        self.actor.train(mode)
        self.critic.train(mode)
        self.determinstic = not mode

    def normalize_states(self, states):
        if states.__class__.__name__ in ["dict", "OrderedDict"]:
            normalized_states = OrderedDict()
            for key, state in states.items():
                normalized_states[key] = self.normalize_factor * state.to(self.device)
            return normalized_states
        else:
            return self.normalize_factor * states.to(self.device)

    def forward(self, states, goal_positions=None) -> tuple([torch.Tensor, torch.Tensor]):
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        if self.use_goal_state and not "goal_position" in states.keys():
            if goal_positions is not None:
                states["goal_position"] = goal_positions.to(states["ball_position"])
            else:
                goal_positions = torch.tensor(self.goal_position).expand_as(states["ball_position"])
                states["goal_position"] = goal_positions.to(states["ball_position"])
        normalized_states = self.normalize_states(states)

        if self.model_name in ["basic", "basic_wo_dropout"]:
            if normalized_states.__class__.__name__ in ["dict", "OrderedDict"]:
                tensors = [v.unsqueeze(0) if v.dim() == 1 else v for v in normalized_states.values()]
                normalized_states = torch.cat(tensors, axis=1).to(self.device)
            elif type(normalized_states) is not torch.Tensor:
                normalized_states = torch.Tensor(normalized_states).to(self.device)

            state_values = self.critic(normalized_states)  # shape: [n_envs,]
            action_logits_vec = self.actor(normalized_states)  # shape: [n_envs, n_actions]
            return (state_values, action_logits_vec)

        elif self.model_name in ["world", "world_gpt"]:
            world_repr = self.world_encoder(normalized_states)
            state_values = self.critic(world_repr)  # shape: [n_envs,]
            action_logits_vec = self.actor(world_repr)  # shape: [n_envs, n_actions]
            return (state_values, action_logits_vec)


    def select_action(
        self, states: torch.Tensor
    ) -> tuple([torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Returns a tuple of the chosen actions and the log-probs of those actions.

        Args:
            x: A batched vector of states.

        Returns:
            actions: A tensor with the actions, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions, with shape [n_steps_per_update, n_envs].
            state_values: A tensor with the state values, with shape [n_steps_per_update, n_envs].
        """
        state_values, action_logits = self.forward(states)
        action_pd = self.dist(logits=action_logits)
        if self.determinstic:
            actions = action_pd.mode()
        else:
            actions = action_pd.sample()
        action_log_probs = action_pd.log_prob(actions)
        entropy = action_pd.entropy()

        return (actions, action_log_probs, state_values, entropy)

    def evaluate_actions(self, states, actions):
        state_values, action_logits = self.forward(states)
        ipdb.set_trace()
        dist = self.dist(logits=action_logits)

        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()

        return state_values, action_log_probs, dist_entropy


class WorldEncoder(nn.Module):
    def __init__(self, feature_shape, hidden_size, dropout=0.0, encoding_type="none", encoding_size=128):
        super().__init__()
        if encoding_type == "none":
            encoding_cls = nn.Identity
            embedding_shape = copy(feature_shape)
        elif encoding_type == "fourier":
            encoding_cls = FourierEncoding
            embedding_shape = [encoding_size + n_feature for n_feature in feature_shape]
        else:
            raise NotImplementedError

        self.world_encoders = nn.ModuleList([
            nn.Sequential(
                encoding_cls(n_feature, n_embedding),
                nn.Linear(n_embedding, hidden_size),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.LayerNorm(hidden_size)
            )
            for n_feature, n_embedding in list(zip(feature_shape, embedding_shape))
        ])
        self.world_gather = nn.Sequential(
                nn.Linear(hidden_size * len(feature_shape), hidden_size),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.LayerNorm(hidden_size)
        )

    def forward(self, input_dict):
        hiddens = []
        for i, (key, inputs) in enumerate(input_dict.items()):
            hidden = self.world_encoders[i](inputs)
            hiddens.append(hidden)
        large_hidden = torch.cat(hiddens, dim=-1)
        world_repr = self.world_gather(large_hidden)
        return world_repr

class WorldTransformerEncoder(nn.Module):
    def __init__(self, feature_shape, hidden_size, n_heads=8, n_layers=1, dropout=0.0, 
                 norm_first=True, activation="gelu"):
        super().__init__()
        self.norm_first = norm_first

        self.position_encoder = nn.Linear(2, hidden_size)
        self.speed_encoder = nn.Linear(2, hidden_size)
        self.direction_encoder = nn.Linear(2, hidden_size)
        self.angular_speed_encoder = nn.Linear(1, hidden_size)

        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden_size, n_heads, 
                                                              dim_feedforward=4*hidden_size,
                                                              dropout=dropout,
                                                              activation=activation,
                                                              norm_first=norm_first) 
                                                for _ in range(n_layers)])
        if norm_first:
            self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, input_dict):
        encodings = []
        for i, (key, inputs) in enumerate(input_dict.items()):
            if re.search(r"angular_speed", key):
                encoding = self.angular_speed_encoder(inputs)
                encodings.append(encoding.unsqueeze(0))
            elif re.search(r"speed", key):
                encoding = self.speed_encoder(inputs)
                encodings.append(encoding.unsqueeze(0))
            elif re.search(r"direction", key):
                encoding = self.direction_encoder(inputs)
                encodings.append(encoding.unsqueeze(0))
            elif re.search(r"position", key):
                encoding = self.position_encoder(inputs)
                encodings.append(encoding.unsqueeze(0))
        hidden = torch.cat(encodings, dim=0)
        
        for layer in self.transformer_layers:
            hidden = layer(hidden)

        if self.norm_first:
            hidden = self.layer_norm(hidden)

        return hidden.mean(dim=0)


class SubdividedActor(nn.Module):
    def __init__(self, action_shape, hidden_size, output_activation="tanh", dropout=0.0):
        super().__init__()

        if output_activation == "none":
            activation_module = nn.Identity()
        elif output_activation == "tanh":
            activation_module = nn.Tanh()
        else:
            raise KeyError("Unknown output activation")

        self.sub_actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_size, action_size),
                activation_module
            )
            for action_size in action_shape
        ])

    def forward(self, world_repr):
        actions = []
        for sub_actor in self.sub_actors:
            action = sub_actor(world_repr)
            actions.append(action)
        actions = torch.cat(actions, dim=-1)
        return actions
