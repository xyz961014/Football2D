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
        model_name: str,
        feature_shape: int,
        action_shape: int,
        hidden_size: int,
        output_activation: str,
        device: torch.device,
        batch_size: int=64,
        n_ppo_epochs: int=4,
        clip_param: float=0.1,
        world_lr: float=1e-3,
        critic_lr: float=5e-3,
        actor_lr: float=1e-3,
        entropy_lr: float=1e-3,
        weight_decay: float=0.0,
        init_scale: float=1.0,
        n_envs=1,
        ent_coef=1.0,
        max_grad_norm=1.0,
        normalize_factor=1.0,
        dropout=0.0,
        encoding_type="none",
        encoding_size=128,
        use_goal_state=False,
        goal_position=(550.0, 0.0)
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__(model_name, feature_shape, action_shape, hidden_size, output_activation,
                         device, init_scale, n_envs, normalize_factor, dropout, encoding_type, encoding_size,
                         use_goal_state, goal_position)
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.batch_size = batch_size
        self.n_ppo_epochs = n_ppo_epochs
        self.clip_param = clip_param

        # define optimizers for actor and critic
        if self.model_name in ["world", "world_gpt"]:
            self.world_params = list(self.world_encoder.parameters())
        self.critic_params = list(self.critic.parameters())
        self.actor_params = list(self.actor.parameters())
        self.entropy_params = list(self.dist.parameters())

        if self.model_name in ["world", "world_gpt"]:
            self.world_optim = optim.Adam(self.world_params, lr=world_lr, weight_decay=weight_decay)
        self.critic_optim = optim.Adam(self.critic_params, lr=critic_lr, weight_decay=weight_decay)
        self.actor_optim = optim.Adam(self.actor_params, lr=actor_lr, weight_decay=weight_decay)
        self.entropy_optim = optim.Adam(self.entropy_params, lr=entropy_lr, weight_decay=weight_decay)

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
        actor_loss = -torch.min(surr1, surr2).mean()

        return (critic_loss, actor_loss, entropy)

    def update_parameters(self, memory):
        advantages = (memory.advantages - memory.advantages.mean()) / (memory.advantages.std() + 1e-5)

        critic_losses = []
        actor_losses = []
        entropies = []
        if self.model_name in ["world", "world_gpt"]:
            keep_state_dict = True
        else:
            keep_state_dict = False

        for epoch in range(self.n_ppo_epochs):
            data_generator = memory.get_ppo_data_generator(self.batch_size, advantages, keep_state_dict)

            for sample in data_generator:

                critic_loss, actor_loss, entropy = self.get_losses(*sample)
                total_loss = critic_loss + actor_loss - self.ent_coef * entropy

                if self.model_name in ["world", "world_gpt"]:
                    self.world_optim.zero_grad()
                self.critic_optim.zero_grad()
                self.actor_optim.zero_grad()
                self.entropy_optim.zero_grad()

                total_loss.backward()

                if self.model_name in ["world", "world_gpt"]:
                    nn.utils.clip_grad_norm_(self.world_params, self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic_params, self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor_params, self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.entropy_params, self.max_grad_norm)

                if self.model_name in ["world", "world_gpt"]:
                    self.world_optim.step()
                self.critic_optim.step()
                self.actor_optim.step()
                self.entropy_optim.step()

                critic_losses.append(critic_loss.item())
                actor_losses.append(actor_loss.item())
                entropies.append(entropy.item())
        
        return (np.mean(critic_losses), np.mean(actor_losses), np.mean(entropies))




