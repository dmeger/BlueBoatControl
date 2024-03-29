import gymnasium as gym
from gymnasium import spaces
import pygame
import math
import numpy as np
import envs
import torch
import argparse
import os
import utils
import TD3


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10., traj_image_count = 0):
    x0 = [0,0,0,0,0,0] 
    # env = gym.make('BlueBoat-v0', X0=x0)
    eval_env = gym.make(env_name, X0=x0)
    # eval_env.seed(seed + 100)
    eval_env.set_img_name("eval", traj_image_count)

    max_action = float(eval_env.action_space.high[0]) # [float(env.action_space.high[i]) for i in range(action_dim)]
    max_action_full = eval_env.action_space.high
    action_scaling_factor = max_action_full / max_action

    max_timesteps = 600
    avg_reward = 0.
    for i in range(eval_episodes):
        # state, done = eval_env.reset(), False
        done = False
        count = 0
        #obs, info = eval_env.reset()
        obs, info = eval_env.reset(seed=seed+10*i)
        state = obs["state"]
        eval_env.render()
        
        while not done:
            action = policy.select_action(np.array(state))
            # action = action * action_scaling_factor # TODO check if necessary? just clip?
            action = action.clip(-max_action_full, max_action_full)
            # state, reward, done, _ = eval_env.step(action)
            # print(action)
            observation, reward, done, truncated, info = eval_env.step(action)
            state = observation["state"]
            boat_pos = state[:2]
            # print(boat_pos)
            eval_env.render()            
            # print("reward: ", reward)
            
            avg_reward += reward
            count += 1
            is_inside = eval_env.is_inside_map(boat_pos[0], boat_pos[1])
            if count >= max_timesteps or (not is_inside):
            # if count >= max_timesteps:
                done = True

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward, eval_env.get_img_count()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="BlueBoat-v0")          # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=5e3, type=int)# Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=1e4, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e8, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.9, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if not os.path.exists("./results/traj_images"):
        os.makedirs("./results/traj_images")

    for file in os.listdir("./results/traj_images"):
        os.remove(os.path.join("./results/traj_images", file))

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # env = gym.make(args.env)
    x0 = [0,0,0,0,0,0] 
    env = gym.make('BlueBoat-v0', X0=x0)

    # Set seeds
    # env.seed(args.seed)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = env.observation_space["state"].shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0]) # [float(env.action_space.high[i]) for i in range(action_dim)]
    max_action_full = env.action_space.high
    action_scaling_factor = max_action_full / max_action
    # print("Max action: ", max_action)
    # print("State dim: ", state_dim)
    # print("Action dim: ", action_dim)
    # print("Max action full: ", max_action_full)

    eval_episodes = 10

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = args.policy_noise * max_action # [args.policy_noise * max_action[i] for i in range(action_dim)]
        kwargs["noise_clip"] = args.noise_clip * max_action # [args.noise_clip * max_action[i] for i in range(action_dim)]
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
# =============================================================================
#     elif args.policy == "OurDDPG":
#         policy = OurDDPG.DDPG(**kwargs)
#     elif args.policy == "DDPG":
#         policy = DDPG.DDPG(**kwargs)
# =============================================================================

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    saved_traj_image_count = 0

    # Evaluate untrained policy
    new_eval, saved_traj_image_count = eval_policy(policy, args.env, args.seed, eval_episodes, saved_traj_image_count)

    evaluations = [new_eval]

    env.set_img_name("train", saved_traj_image_count)

    # state, done = env.reset(), False
    done = False
    obs, info = env.reset(seed=args.seed)
    state = obs["state"]
    env.render()
    
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    episode_max_timesteps = 700

    # print("Max action: ", max_action)
    # print("State dim: ", state_dim)
    # print("Action dim: ", action_dim)
    # print("Max action full: ", max_action_full)

    for t in range(int(args.max_timesteps)):
        
        episode_timesteps += 1

        # Select action randomly or according to policy
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
            )

            # print("Policy raw action: ", action)

            action+= np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            
            action = action * action_scaling_factor
            
            action = action.clip(-max_action_full, max_action_full)
            # action = action.clip(-max_action, max_action)

            # print("Action: ", action)

        # Perform action
        observation, reward, done, truncated, info = env.step(action)
        next_state = observation["state"]
        env.render()

        done = done or episode_timesteps >= episode_max_timesteps
        
        done_bool = float(done) # if episode_timesteps < env._max_episode_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # Train agent after collecting sufficient data
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done: 
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            # state, done = env.reset(), False
            done = False
            obs, info = env.reset()
            state = obs["state"]
            
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1 

        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            saved_traj_image_count = env.get_img_count() 

            new_eval, saved_traj_image_count = eval_policy(policy, args.env, args.seed, eval_episodes, saved_traj_image_count)

            env.set_img_name("train", saved_traj_image_count)

            evaluations.append(new_eval)
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model: policy.save(f"./models/{file_name}")
