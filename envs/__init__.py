import gymnasium as gym
from .blueboat_env import BlueBoat
from .blueboat_sbenv import SBBlueBoat
from gymnasium.envs.registration import register

register(
    id='BlueBoat-v0',
    entry_point='envs:BlueBoat',
    max_episode_steps=300
    )