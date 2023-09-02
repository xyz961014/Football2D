import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import deque, OrderedDict
import rl.utils as utils

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

    def after_update(self, final_state):
        self.states.clear()
        self.actions = torch.zeros(self.memory_size, self.n_envs, self.n_actions, device=self.device)
        self.value_preds = torch.zeros(self.memory_size + 1, self.n_envs, device=self.device)
        self.rewards = torch.zeros(self.memory_size, self.n_envs, device=self.device)
        self.action_log_probs = torch.zeros(self.memory_size, self.n_envs, device=self.device)
        self.masks = torch.ones(self.memory_size, self.n_envs, device=self.device)

        self.states.append(final_state)

    def compute_returns_and_advantages(self, gamma, lam, next_state_value):

        T = len(self.rewards)
        self.advantages = torch.zeros(T, self.value_preds.shape[1], device=self.device)
        self.returns = torch.zeros(T, self.value_preds.shape[1], device=self.device)

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

    ######## HER ########

    def her_sample_new_goals(self, k_samples, strategy="future"):
        assert len(self.states) == self.memory_size + 1
        states_with_new_goals = []
        if strategy == "future":
            for step in range(len(self.states)):
                state = utils.concat_dict_tensors([utils.clone_dict_tensor(self.states[step]) 
                                                   for _ in range(k_samples)])
                goal_steps = np.random.choice(range(step, len(self.states)), size=k_samples)
                step_goal_positions = []
                for goal_step in goal_steps:
                    step_goal_position = self.states[goal_step]["ball_position"].detach().clone()
                    step_goal_positions.append(step_goal_position)
                step_goal_positions = torch.cat(step_goal_positions, dim=0)
                state["goal_position"] = step_goal_positions.to(self.device)
                states_with_new_goals.append(state)
        elif strategy == "final":
            goal_step = len(self.states) - 1
            goal_position = self.states[goal_step]["ball_position"].detach().clone()
            for step in range(len(self.states)):
                state = utils.concat_dict_tensors([utils.clone_dict_tensor(self.states[step]) 
                                                   for _ in range(k_samples)])
                state["goal_position"] = goal_position.repeat(self.n_envs, 1).to(self.device)
                states_with_new_goals.append(state)
        else:
            raise ValueError("Unknown strategy")

        return states_with_new_goals

    def her_evaluate_transitions(self, states_with_new_goals, agent, tolerance=10.0, batch_size=1024):
        states = utils.concat_dict_tensors(states_with_new_goals[:-1])
        k_samples = states["goal_position"].size(0) // (self.n_envs * self.memory_size)
        actions = self.actions.repeat(1, k_samples, 1)
        actions = actions.reshape(-1, *self.actions.shape[2:])
        rewards = self.rewards.repeat(1, k_samples)
        rewards = rewards.reshape(-1, *self.rewards.shape[2:])

        with torch.no_grad():
            num_chunks = (actions.size(0) + batch_size - 1) // batch_size
            state_chunks = utils.chunk_dict_tensor(states, num_chunks, dim=0)
            action_chunks = torch.chunk(actions, num_chunks, dim=0)

            state_values = []
            action_log_probs = []
            for state_chunk, action_chunk in list(zip(state_chunks, action_chunks)):
                state_values_chunk, action_log_probs_chunk, _ = agent.evaluate_actions(state_chunk, action_chunk)
                state_values.append(state_values_chunk)
                action_log_probs.append(action_log_probs_chunk)
            state_values = torch.cat(state_values, dim=0)
            action_log_probs = torch.cat(action_log_probs, dim=0)

        new_rewards = (states["goal_position"] - states["ball_position"]).norm(dim=1) < tolerance
        new_rewards = new_rewards.to(self.rewards) - 1.0 # -1/0 reward
        new_rewards  = new_rewards + rewards

        state_values = state_values.reshape(self.memory_size, -1)
        action_log_probs = action_log_probs.reshape(self.memory_size, -1)
        new_rewards = new_rewards.reshape(self.memory_size, -1)

        return state_values, action_log_probs, new_rewards

    def her_insert(self, states_with_new_goals, her_action_log_probs, her_state_value_preds, her_rewards):
        k_samples = her_rewards.size(1) // self.n_envs
        for step, state in enumerate(states_with_new_goals):
            self.states[step] = utils.concat_dict_tensors((self.states[step], state))

        padding = (0, 0, 0, self.value_preds.shape[0] - her_state_value_preds.shape[0])
        padded_state_value_preds = F.pad(her_state_value_preds, padding, mode="constant", value=0.0)
        self.value_preds = torch.cat((self.value_preds, padded_state_value_preds), dim=1)

        self.actions = self.actions.repeat(1, k_samples + 1, 1)
        self.masks = self.masks.repeat(1, k_samples + 1)

        self.action_log_probs = torch.cat((self.action_log_probs, her_action_log_probs), dim=1)
        self.rewards = torch.cat((self.rewards, her_rewards), dim=1)


