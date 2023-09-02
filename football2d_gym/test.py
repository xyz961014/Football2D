import gym
import argparse
import football2d
import numpy as np
import time
import ipdb


parser = argparse.ArgumentParser()
parser.add_argument("--one_on_one", action="store_true")
args = parser.parse_args()

if args.one_on_one:
    env = gym.make("football2d/OneOnOneMatch", render_mode="human")
    state, info = env.reset()
    step = 0
    terminated = False
    truncated = False
    try:
        while True:
            home_action = np.array([[0., 0., 
                                     0., 0.,
                                     0.,
                                   ]])
            away_action = np.array([[0., 0.5, 
                                     1., 0.,
                                     -0.1,
                                   ]])
            action = {
                    "home_action": home_action,
                    "away_action": away_action
                     }
            observation, reward, terminated, truncated, info = env.step({})
            step += 1
    except KeyboardInterrupt:
        ipdb.set_trace()
else:
    env = gym.make('football2d/SelfTraining-v2', render_mode="human", 
                   learn_to_kick=True,
                   randomize_position=False,
                   ball_position=(0, 20))
    state, info = env.reset()
    step = 0
    terminated = False
    truncated = False
    try:
        #while not (terminated or truncated):
        while True:
            if step == 0:
                action = np.array([0., 0., 
                                   0., 0.,
                                   0.,
                                   ])
            else:
                action = np.array([1., 0., 
                                   1., 0.,
                                   -0.1,
                                   ])
            observation, reward, terminated, truncated, info = env.step(action)
            step += 1
    except KeyboardInterrupt:
        ipdb.set_trace()

