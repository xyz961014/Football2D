import numpy as np
import torch
import gym
import ipdb


class EnvPyTorchWrapper(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.device = device

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.convert_nparray_to_tensor(obs)
        return obs, info

    def step(self, action):
        action = action.cpu().numpy()
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.convert_nparray_to_tensor(obs)
        return obs, reward, terminated, truncated, info

    def convert_nparray_to_tensor(self, array_or_dict):
        if array_or_dict.__class__.__name__ in ["OrderedDict", "dict"]:
            for key, value in array_or_dict.items():
                array_or_dict[key] = self.convert_nparray_to_tensor(value)
        else:
            array_or_dict = torch.from_numpy(array_or_dict).float().to(self.device)
        return array_or_dict


class VectorEnvPyTorchWrapper(EnvPyTorchWrapper):
    def __init__(self, env, device):
        super().__init__(env, device)

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.env.step_async(actions)

    def step_wait(self):
        obs, reward, terminated, truncated, info = self.env.step_wait()
        obs = self.convert_nparray_to_tensor(obs)
        reward = self.convert_nparray_to_tensor(reward)
        terminated = self.convert_nparray_to_tensor(terminated)
        truncated = self.convert_nparray_to_tensor(truncated)
        return obs, reward, terminated, truncated, info


class PyTorchRecordEpisodeStatistics(gym.wrappers.RecordEpisodeStatistics):
    def __init__(self, env, deque_size, device):
        super().__init__(env, deque_size)
        self.device = device

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float).to(self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        return observations




