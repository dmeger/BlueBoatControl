import gym
from gym import spaces
import pygame
import math
import numpy as np
import envs
# =============================================================================
# from scipy.integrate import ode
# import clock
# from blueboat_env import BlueBoat
# =============================================================================

x0 = [0,0,0,0,0,0] 
env = gym.make('BlueBoat-v0', X0=x0)
obs, state, reward, info = env.reset()
env.render()

while True:
    action = env.action_space.sample()
    obs, reward, done, _, _= env.step(action)
    env.render()
    print("reward:", reward)
    # print('\n')
    
    #pygame.time.wait(5)
