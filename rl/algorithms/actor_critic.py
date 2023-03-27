import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import ipdb


from pathlib import Path
parent_dir = Path(__file__).absolute().parent.parent.parent
sys.path.append(os.path.abspath(parent_dir))

from rl.utils import ScaleParameterizedNormal

class ActorCritic(nn.Module):
    def __init__(self, n_features, n_actions, hidden_size, output_activation, 
                       device, init_scale, n_envs, normalize_factor=1.0, dropout=0.0):
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.normalize_factor = normalize_factor

        self.determinstic = False

        self.dist = ScaleParameterizedNormal(n_actions=n_actions, init_scale=init_scale).to(self.device)

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
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

    def eval(self):
        super().eval()
        self.actor.eval()
        self.critic.eval()
        self.determinstic = True

    def train(self, mode=True):
        super().train(mode)
        self.actor.train(mode)
        self.critic.train(mode)
        self.determinstic = not mode

    def normalize_states(self, states):
        return states * self.normalize_factor

    def forward(self, states: np.ndarray) -> tuple([torch.Tensor, torch.Tensor]):
        """
        Forward pass of the networks.

        Args:
            x: A batched vector of states.

        Returns:
            state_values: A tensor with the state values, with shape [n_envs,].
            action_logits_vec: A tensor with the action logits, with shape [n_envs, n_actions].
        """
        if states.__class__.__name__ in ["dict", "OrderedDict"]:
            tensors = [v.unsqueeze(0) if v.dim() == 1 else v for v in states.values()]
            states = torch.cat(tensors, axis=1).to(self.device)
        elif type(states) is not torch.Tensor:
            states = torch.Tensor(states).to(self.device)

        normalized_states = self.normalize_states(states)
        state_values = self.critic(normalized_states)  # shape: [n_envs,]
        action_logits_vec = self.actor(normalized_states)  # shape: [n_envs, n_actions]
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
        dist = self.dist(logits=action_logits)

        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()

        return state_values, action_log_probs, dist_entropy


