import numpy as np
import torch
from collections import deque

import ipdb

class TrainingMemory(object):
    def __init__(self, n_steps_per_update, n_envs, obs_shape, action_shape, device):
        super().__init__()

        self.n_steps_per_update = n_steps_per_update
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device

        self.value_preds = torch.zeros(n_steps_per_update + 1, n_envs, device=self.device)
        self.rewards = torch.zeros(n_steps_per_update, n_envs, device=self.device)
        self.action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=self.device)
        self.masks = torch.ones(n_steps_per_update, n_envs, device=self.device)
        self.actions = torch.zeros(n_steps_per_update, n_envs, action_shape, device=self.device)
        self.states = deque(maxlen=n_steps_per_update + 1)

        #self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.step = 0

    def concat_states(self):
        states = [torch.cat([v.unsqueeze(0) if v.dim() == 1 else v for v in states.values()]) 
                  if states.__class__.__name__ in ["dict", "OrderedDict"] else states 
                  for states in list(self.states)[:-1]]
        states = torch.cat(states)
        return states.reshape(self.n_steps_per_update, self.n_envs, self.obs_shape)

    def insert(self, states, actions, action_log_probs, state_value_preds, rewards, masks):

        self.states.append(states)

        self.actions[self.step] = actions
        self.value_preds[self.step] = torch.squeeze(state_value_preds)
        self.rewards[self.step] = rewards
        self.action_log_probs[self.step] = action_log_probs
        self.masks[self.step] = masks

        self.step = (self.step + 1) % self.n_steps_per_update

    def after_update(self):
        final_state = self.states[-1]
        
        self.states.clear()
        self.actions = torch.zeros(self.n_steps_per_update, self.n_envs, self.action_shape, device=self.device)
        self.value_preds = torch.zeros(self.n_steps_per_update + 1, self.n_envs, device=self.device)
        self.rewards = torch.zeros(self.n_steps_per_update, self.n_envs, device=self.device)
        self.action_log_probs = torch.zeros(self.n_steps_per_update, self.n_envs, device=self.device)
        self.masks = torch.ones(self.n_steps_per_update, self.n_envs, device=self.device)

        self.states.append(final_state)

    def compute_returns_and_advantages(self, gamma, lam, next_state_value):

        T = len(self.rewards)
        self.advantages = torch.zeros(T, self.n_envs, device=self.device)
        self.returns = torch.zeros(T, self.n_envs, device=self.device)

        self.value_preds[-1] = next_state_value
        # compute the advantages using GAE
        gae = 0.0
        for t in reversed(range(T)):
            td_error = (
                self.rewards[t] + gamma * self.masks[t] * self.value_preds[t + 1] - self.value_preds[t]
            )
            gae = td_error + gamma * lam * self.masks[t] * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.value_preds[t]

        return self.returns, self.advantages

