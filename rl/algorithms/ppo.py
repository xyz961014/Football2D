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
from rl.algorithms.actor_critic import ActorCritic

class PPO(ActorCritic):
    def __init__(
        self,
        n_features: int,
        n_actions: int,
        hidden_size: int,
        output_activation: str,
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
        super().__init__(n_features, n_actions, hidden_size, output_activation,
                         device, init_scale, n_envs, normalize_factor)
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.batch_size = batch_size
        self.n_ppo_epochs = n_ppo_epochs
        self.clip_param = clip_param

        # define optimizers for actor and critic
        self.critic_params = list(self.critic.parameters())
        self.actor_params = list(self.actor.parameters())
        if train_scale and hasattr(self.dist, "parameters"):
            self.actor_params.extend(list(self.dist.parameters()))
        self.critic_optim = optim.Adam(self.critic_params, lr=critic_lr)
        self.actor_optim = optim.Adam(self.actor_params, lr=actor_lr)

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




