import gym
from gym.spaces import Box
import numpy as np
import ipdb


class RelativeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, obs):
        if "player_position" in obs.keys():
            if "ball_position" in obs.keys():
                obs["ball_position"] = obs["ball_position"] - obs["player_position"]
        if "player_speed" in obs.keys():
            if "ball_speed" in obs.keys():
                obs["ball_speed"] = obs["ball_speed"] - obs["player_speed"]
        return obs
