import gymnasium as gym
from gymnasium import spaces
import pygame
import math
import numpy as np
import envs

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

x0 = [0,0,0,0,0,0] 
env = gym.make('BlueBoat-v0', X0=x0)

# check_env(env, warn=True)
# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3("MultiInputPolicy", env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=100, log_interval=10)
# model.save("td3_pendulum")
vec_env = model.get_env()
# del model # remove to demonstrate saving and loading
# model = TD3.load("td3_pendulum")

obs, info = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, _, _ = vec_env.step(action)
    vec_env.render("human")

