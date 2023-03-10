import numpy as np
import torch
import ipdb


class AuxiliaryRewardManager(object):
    def __init__(self, device, reward_type="default"):
        super().__init__()
        self.device = device
        self.reward_type = reward_type

    def __call__(self, info):
        raise NotImplementedError


class AuxiliaryRewardManager_SelfTraining_v0(AuxiliaryRewardManager):
    def __init__(self, device, reward_type="default"):
        super().__init__(device, reward_type)

    def __call__(self, infos):
        if self.reward_type == "default":
            aux_rewards = -(infos["distance_to_ball"] + infos["distance_to_goal"]) * 1e-6
            #aux_rewards = -(infos["distance_to_ball"]) * 1e-6
            return torch.from_numpy(aux_rewards).to(self.device)
        else:
            raise NotImplementedError

class AuxiliaryRewardManager_SelfTraining_v1(AuxiliaryRewardManager):
    def __init__(self, device, reward_type="default"):
        super().__init__(device, reward_type)

    def __call__(self, info):
        if self.reward_type == "default":
            pass
        else:
            raise NotImplementedError


class AuxiliaryRewardManager_SelfTraining_v2(AuxiliaryRewardManager):
    def __init__(self, device, reward_type="default"):
        super().__init__(device, reward_type)

    def __call__(self, info):
        if self.reward_type == "default":
            pass
        else:
            raise NotImplementedError


def get_auxiliary_reward_manager(env_name, device, reward_type="default"):
    auxiliary_reward_manager_map = {
            "SelfTraining-v0": AuxiliaryRewardManager_SelfTraining_v0,
            "SelfTraining-v1": AuxiliaryRewardManager_SelfTraining_v1,
            "SelfTraining-v2": AuxiliaryRewardManager_SelfTraining_v2
            }
    auxiliary_reward_manager = auxiliary_reward_manager_map[env_name](device, reward_type)
    return auxiliary_reward_manager
