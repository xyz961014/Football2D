import gym
import football2d
import numpy as np
import time
import ipdb
env = gym.make('football2d/SelfTraining-v0', render_mode="human", randomize_position=True)
state = env.reset()
step = 0
terminated = False
truncated = False
try:
    while True:
        action = np.random.uniform(-1.0, 1.0, (2, 2))
        #action = np.array([[1., 1.], 
        #                   [1., -1.]])
        observation, reward, terminated, truncated, info = env.step(action)
        step += 1
except KeyboardInterrupt:
    ipdb.set_trace()

