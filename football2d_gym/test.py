import gym
import football2d
import numpy as np
import time
import ipdb
env = gym.make('football2d/SelfTraining-v2', render_mode="human", randomize_position=False)
state = env.reset()
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
                               -0.2,
                               ])
        observation, reward, terminated, truncated, info = env.step(action)
        step += 1
except KeyboardInterrupt:
    ipdb.set_trace()

