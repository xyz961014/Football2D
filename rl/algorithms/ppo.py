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

class PPO(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        hidden_size: int,
        device: torch.device,
        batch_size: int,
        n_ppo_epochs: int,
        clip_param: float,
        critic_lr: float,
        actor_lr: float,
        init_scale: float,
        n_envs: int,
        ent_coef=1.0,
        max_grad_norm=1.0,
        train_scale=True,
        normalize_factor=1.0
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.normalize_factor = normalize_factor

        self.batch_size = batch_size
        self.n_ppo_epochs = n_ppo_epochs
        self.clip_param = clip_param

        self.determinstic = False

        self.dist = ScaleParameterizedNormal(n_actions=n_actions, init_scale=init_scale).to(self.device)

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
            #nn.Tanh() # estimate action logits
        ]

        # define actor and critic networks
        self.critic = nn.Sequential(*critic_layers).to(self.device)
        self.actor = nn.Sequential(*actor_layers).to(self.device)

        # define optimizers for actor and critic
        self.critic_params = list(self.critic.parameters())
        self.actor_params = list(self.actor.parameters())
        if train_scale and hasattr(self.dist, "parameters"):
            self.actor_params.extend(list(self.dist.parameters()))
        self.critic_optim = optim.Adam(self.critic_params, lr=critic_lr)
        self.actor_optim = optim.Adam(self.actor_params, lr=actor_lr)

    def eval(self):
        super().eval()
        self.actor.eval()
        self.critic.eval()
        self.determinstic = True

    def train(self, mode=True):
        super().train(mode)
        self.actor.train()
        self.critic.train()
        self.determinstic = False

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

    def get_losses(self,
        states_batch,
        actions_batch,
        value_preds_batch,
        returns_batch,
        masks_batch,
        old_action_log_probs_batch,
        advantages_batch
    ):
        state_values, action_log_probs, entropy = self.evaluate_actions(states_batch, actions_batch)
        values = state_values.squeeze()

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
        value_losses = (values - returns_batch).pow(2)
        value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
        critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        actor_loss = -torch.min(surr1, surr2).mean() - self.ent_coef * entropy

        return (critic_loss, actor_loss, entropy)

    def update_parameters(self, memory):
        advantages = (memory.advantages - memory.advantages.mean()) / (memory.advantages.std() + 1e-5)

        critic_losses = []
        actor_losses = []
        entropies = []
        for epoch in range(self.n_ppo_epochs):
            data_generator = memory.get_ppo_data_generator(self.batch_size, advantages)

            for sample in data_generator:

                critic_loss, actor_loss, entropy = self.get_losses(*sample)

                self.critic_optim.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                self.critic_optim.step()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                self.actor_optim.step()

                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss.item())
                entropies.append(entropy.item())
        
        return (np.mean(critic_losses), np.mean(actor_losses), np.mean(entropies))




