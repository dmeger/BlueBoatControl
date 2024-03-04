import gymnasium as gym
from gymnasium import spaces
import pygame
import math
import numpy as np
import envs
# from envs.blueboat_env import BlueBoat
from envs.blueboat_sbenv import SBBlueBoat

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import TD3, PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

x0 = [0,0,0,0,0,0] 
# env = gym.make('BlueBoat-v0', X0=x0)
env = SBBlueBoat(X0=x0)

# check_env(env, warn=True, skip_render_check=False)

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# env.render()
# model = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=1)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=10)
# model.save("td3_pendulum")
vec_env = model.get_env()
# del model # remove to demonstrate saving and loading
# model = TD3.load("td3_pendulum")

obs = vec_env.reset()
while True:
    # print("obs: ", obs)   # OrderedDict([('state', array([[ 3.410268,3.2651038,-1.4433166,0.05670824,0.32463634.-0.17810197]], dtype=float32))])
    action, _states = model.predict(obs)
    
    # print("action: ", action)     # action:  [[1.6387875e+00 3.7512183e-04]]
    # print("states: ", _states)    # state: None
    
    obs, rewards, dones, infos = vec_env.step(action)
    
    # print("obs: ", obs)
    print("rwd: ", rewards)
    # print("done: ", dones)
    # print("info: ", infos)
    
    vec_env.render("rgb_array")

