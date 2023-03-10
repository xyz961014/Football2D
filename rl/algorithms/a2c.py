import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim

import ipdb


curr_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(curr_path, "..", ".."))
from rl.utils import ScaleParameterizedNormal

class A2C(nn.Module):
    """
    (Synchronous) Advantage Actor-Critic agent class

    Args:
        n_features: The number of features of the input state.
        n_actions: The number of actions the agent can take.
        device: The device to run the computations on (running on a GPU might be quicker for larger Neural Nets,
                for this code CPU is totally fine).
        critic_lr: The learning rate for the critic network (should usually be larger than the actor_lr).
        actor_lr: The learning rate for the actor network.
        n_envs: The number of environments that run in parallel (on multiple CPUs) to collect experiences.
    """

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        hidden_size: int,
        device: torch.device,
        critic_lr: float,
        actor_lr: float,
        init_scale: float,
        n_envs: int,
        train_scale=True
    ) -> None:
        """Initializes the actor and critic networks and their respective optimizers."""
        super().__init__()
        self.device = device
        self.n_envs = n_envs

        self.determinstic = False

        self.dist = ScaleParameterizedNormal(shape=(n_envs, n_actions), init_scale=init_scale).to(self.device)

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
        critic_params = list(self.critic.parameters())
        actor_params = list(self.actor.parameters())
        if train_scale and hasattr(self.dist, "parameters"):
            actor_params.extend(list(self.dist.parameters()))
        self.critic_optim = optim.RMSprop(critic_params, lr=critic_lr)
        self.actor_optim = optim.RMSprop(actor_params, lr=actor_lr)

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
        return states * 1e-3

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

    def get_losses(
        self,
        memory,
        ent_coef: float,
    ) -> tuple([torch.Tensor, torch.Tensor]):
        """
        Computes the loss of a minibatch (transitions collected in one sampling phase) for actor and critic
        using Generalized Advantage Estimation (GAE) to compute the advantages (https://arxiv.org/abs/1506.02438).

        Args:
            rewards: A tensor with the rewards for each time step in the episode, with shape [n_steps_per_update, n_envs].
            action_log_probs: A tensor with the log-probs of the actions taken at each time step in the episode, with shape [n_steps_per_update, n_envs].
            value_preds: A tensor with the state value predictions for each time step in the episode, with shape [n_steps_per_update, n_envs].
            masks: A tensor with the masks for each time step in the episode, with shape [n_steps_per_update, n_envs].
            gamma: The discount factor.
            lam: The GAE hyperparameter. (lam=1 corresponds to Monte-Carlo sampling with high variance and no bias,
                                          and lam=0 corresponds to normal TD-Learning that has a low variance but is biased
                                          because the estimates are generated by a Neural Net).
            device: The device to run the computations on (e.g. CPU or GPU).

        Returns:
            critic_loss: The critic loss for the minibatch.
            actor_loss: The actor loss for the minibatch.
        """
        #advantages = memory.advantages
        #action_log_probs = memory.action_log_probs
        #entropy = self.dist(memory.actions[-1]).entropy()

        states = memory.concat_states()
        state_values, action_log_probs, entropy = self.evaluate_actions(states, memory.actions)
        value_preds = state_values.squeeze()

        advantages = memory.returns - value_preds

        # calculate the loss of the minibatch for actor and critic
        critic_loss = advantages.pow(2).mean()

        # give a bonus for higher entropy to encourage exploration
        actor_loss = (
            -(advantages.detach() * action_log_probs).mean() - ent_coef * entropy.mean()
        )
        return (critic_loss, actor_loss)

    def update_parameters(
        self, critic_loss: torch.Tensor, actor_loss: torch.Tensor
    ) -> None:
        """
        Updates the parameters of the actor and critic networks.

        Args:
            critic_loss: The critic loss.
            actor_loss: The actor loss.
        """
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()


