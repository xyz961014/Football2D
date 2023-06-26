import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import deque, OrderedDict

import ipdb

class TrainingMemory(object):
    def __init__(self, memory_size, n_envs, obs_shape, action_shape, device):
        super().__init__()

        self.memory_size = memory_size
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.n_features = sum(obs_shape) if type(obs_shape) in [tuple, list] else obs_shape
        self.action_shape = action_shape
        self.n_actions = sum(action_shape) if type(action_shape) in [tuple, list] else action_shape
        self.device = device

        self.value_preds = torch.zeros(memory_size + 1, n_envs, device=self.device)
        self.rewards = torch.zeros(memory_size, n_envs, device=self.device)
        self.action_log_probs = torch.zeros(memory_size, n_envs, device=self.device)
        self.masks = torch.ones(memory_size, n_envs, device=self.device)
        self.actions = torch.zeros(memory_size, n_envs, self.n_actions, device=self.device)
        self.states = deque(maxlen=memory_size + 1)

        #self.returns = torch.zeros(num_steps + 1, num_processes, 1)

        self.step = 0

    def concat_states(self, keep_state_dict=False):
        if keep_state_dict:
            states = OrderedDict()
            for state_dict in list(self.states)[:-1]:
                for key, value in state_dict.items():
                    if key in states.keys():
                        states[key].append(value.unsqueeze(0))
                    else:
                        states[key] = [value.unsqueeze(0)]
            for key, value in states.items():
                states[key] = torch.cat(value, dim=0)
            return states
        else:
            states = [torch.cat([v.unsqueeze(0) if v.dim() == 1 else v for v in states.values()]).reshape(len(states), self.n_envs, -1).permute(1, 0, 2).reshape(self.n_envs, self.n_features) 
                      if states.__class__.__name__ in ["dict", "OrderedDict"] else states 
                      for states in list(self.states)[:-1]]
            states = torch.cat(states)
            return states.reshape(self.memory_size, self.n_envs, self.n_features)

    def insert(self, states, actions, action_log_probs, state_value_preds, rewards, masks):

        self.states.append(states)

        self.actions[self.step] = actions
        self.value_preds[self.step] = torch.squeeze(state_value_preds)
        self.rewards[self.step] = rewards
        self.action_log_probs[self.step] = action_log_probs
        self.masks[self.step] = masks

        self.step = (self.step + 1) % self.memory_size

    def after_update(self):
        final_state = self.states[-1]
        
        self.states.clear()
        self.actions = torch.zeros(self.memory_size, self.n_envs, self.n_actions, device=self.device)
        self.value_preds = torch.zeros(self.memory_size + 1, self.n_envs, device=self.device)
        self.rewards = torch.zeros(self.memory_size, self.n_envs, device=self.device)
        self.action_log_probs = torch.zeros(self.memory_size, self.n_envs, device=self.device)
        self.masks = torch.ones(self.memory_size, self.n_envs, device=self.device)

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

    def get_ppo_data_generator(self, batch_size, advantages=None, keep_state_dict=False):

        sample_size = self.n_envs * (len(self.states) - 1)
        assert batch_size <= sample_size
        sampler = BatchSampler(SubsetRandomSampler(range(sample_size)), batch_size, drop_last=True)

        def prepare_data_batch(data, inds):
            data_batch = data.reshape(-1, *data.shape[2:])[inds]
            return data_batch

        def prepare_data_batch_dict(data_dict, inds):
            data_batch_dict = OrderedDict()
            for key, data in data_dict.items():
                data_batch = data.reshape(-1, *data.shape[2:])[inds]
                data_batch_dict[key] = data_batch
            return data_batch_dict

        # get states tensor
        states = self.concat_states(keep_state_dict)

        for indices in sampler:
            if keep_state_dict:
                states_batch = prepare_data_batch_dict(states, indices)
            else:
                states_batch = prepare_data_batch(states, indices)
            actions_batch = prepare_data_batch(self.actions, indices)
            value_preds_batch = prepare_data_batch(self.value_preds, indices)
            returns_batch = prepare_data_batch(self.returns, indices)
            masks_batch = prepare_data_batch(self.masks, indices)
            old_action_log_probs_batch = prepare_data_batch(self.action_log_probs, indices)
            if advantages is None:
                advantages_batch = None
            else:
                advantages_batch = prepare_data_batch(advantages, indices)

            yield (states_batch,
                   actions_batch,
                   value_preds_batch,
                   returns_batch,
                   masks_batch,
                   old_action_log_probs_batch,
                   advantages_batch)


