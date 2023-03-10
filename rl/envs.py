import numpy as np
import time
import torch
import gym
from gym.wrappers.record_episode_statistics import add_vector_episode_statistics
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
    def __init__(self, env, deque_size, device, auxiliary_reward_manager=None):
        super().__init__(env, deque_size)
        self.device = device
        self.auxiliary_reward_manager = auxiliary_reward_manager

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float).to(self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int).to(self.device)
        return observations

    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = self.env.step(action)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        ###############################################################
        # This part is the only change of the function
        # Add auxiliary reward
        if self.auxiliary_reward_manager is not None:
            rewards += self.auxiliary_reward_manager(infos)
        ###############################################################
        self.episode_returns += rewards
        self.episode_lengths += 1
        if not self.is_vector_env:
            terminateds = [terminateds]
            truncateds = [truncateds]
        terminateds = list(terminateds)
        truncateds = list(truncateds)

        for i in range(len(terminateds)):
            if terminateds[i] or truncateds[i]:
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "episode": {
                        "r": episode_return,
                        "l": episode_length,
                        "t": round(time.perf_counter() - self.t0, 6),
                    }
                }
                if self.is_vector_env:
                    infos = add_vector_episode_statistics(
                        infos, episode_info["episode"], self.num_envs, i
                    )
                else:
                    infos = {**infos, **episode_info}
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return (
            observations,
            rewards,
            terminateds if self.is_vector_env else terminateds[0],
            truncateds if self.is_vector_env else truncateds[0],
            infos,
        )




