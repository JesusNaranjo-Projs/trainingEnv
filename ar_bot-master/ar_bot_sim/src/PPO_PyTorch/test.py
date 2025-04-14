import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from PPO_PyTorch.PPO import PPO, Random


#################################### Testing ###################################
def test(env_class, model_name=None, path=None, render=False, rand=False, baseline=False):
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "ARBotGymEnv"
    has_continuous_action_space = True
    max_ep_len = 1500           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 100    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################

    env = env_class(gui=render, path="ar_bot_gym/", max_timesteps=max_ep_len)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    # initialize a PPO agent
    if not baseline:
        ppo_agent1 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    else:
        ppo_agent1 = Random(random_seed)
    
    if not rand and not baseline:
        ppo_agent2 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    else:
        ppo_agent2 = Random(random_seed)
    

    if path == None:
        directory = "trained_models"
        directory = directory + '/' + env_name + '/'
    else:
        directory = path + "/"

    if model_name == None:
        checkpoint_path1 = directory + "PPO1_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
        checkpoint_path2 = directory + "PPO2_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    else:
        checkpoint_path1 = directory + "1" + model_name
        checkpoint_path2 = directory + "2" + model_name

    if not baseline:
        ppo_agent1.load(checkpoint_path1)
    
    if not rand and not baseline:
        ppo_agent2.load(checkpoint_path2)

    print("--------------------------------------------------------------------------------------------")

    test_running_rewards1 = []
    num_wins1 = 0 # wins are defined as having a higher total reward
    num_goals1 = 0
    
    test_running_rewards2 = []
    num_wins2 = 0 # wins are defined as having a higher total reward
    num_goals2 = 0
    
    num_timesteps = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward1 = 0
        ep_reward2 = 0
        state, _, _  = env.reset()

        for t in range(1, max_ep_len+1):
            action1 = ppo_agent1.select_action(state)
            action2 = ppo_agent2.select_action(state)
            state, reward1, reward2, done, goal_scorer = env.step_both(action1, action2)
            ep_reward1 += reward1
            ep_reward2 += reward2
            
            num_timesteps += 1

            if render:
                env.render()
                time.sleep(frame_delay)

            if done:
                break

        # clear buffer
        if not baseline:
            ppo_agent1.buffer.clear()
        if not rand and not baseline:
            ppo_agent2.buffer.clear()
            
        if ep_reward1 > ep_reward2:
            num_wins1 += 1
        elif ep_reward2 > ep_reward1:
            num_wins2 += 1
        
        if goal_scorer == 1:
            num_goals1 += 1
        elif goal_scorer == 2:
            num_goals2 += 1

        test_running_rewards1.append(ep_reward1)
        test_running_rewards2.append(ep_reward2)
        print('Episode: {} \t\t Reward 1: {}'.format(ep, round(ep_reward1, 2)))
        print('Episode: {} \t\t Reward 2: {}'.format(ep, round(ep_reward2, 2)))

    env.close()

    print("============================================================================================")

    
    avg_test_reward1 = np.mean(test_running_rewards1)
    avg_test_reward2 = np.mean(test_running_rewards2)
    avg_test_reward1 = round(avg_test_reward1, 2)
    avg_test_reward2 = round(avg_test_reward2, 2)
    std1 = round(np.std(test_running_rewards1), 2)
    std2 = round(np.std(test_running_rewards2), 2)

    print(f"average test reward 1: {avg_test_reward1}")
    print(f"standard deviation 1: {std1}")
    print(f"average test reward 2: {avg_test_reward2}")
    print(f"standard deviation 2: {std2}")
    print(f"winrate for agent 1: {num_wins1 / total_test_episodes}")
    print(f"num goals scored for agent 1: {num_goals1}")
    print(f"winrate for agent 2: {num_wins2 / total_test_episodes}")
    print(f"num goals scored for agent 2: {num_goals2}")
    print(f"average timesteps per episode: {num_timesteps / total_test_episodes}")

    print("============================================================================================")

    return avg_test_reward1, avg_test_reward2


if __name__ == '__main__':

    test()
