import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from PPO_PyTorch.PPO import PPO

################################### Training ###################################
def train(env_class, model_name=None):
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "ARBotGymEnv"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 5000                   # max timesteps in one episode
    max_training_timesteps = int(1e7)   # break training loop if timeteps > max_training_timesteps original: 3e6

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = env_class(gui=False, path = "ar_bot_gym/", max_timesteps=max_ep_len)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + 'PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "trained_models"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path1 = directory + "PPO1_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    checkpoint_path2 = directory + "PPO2_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    if model_name == None:
        ppo_agent1 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
        ppo_agent2 = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)
    else:
        ppo_agent1 = PPO.load(model_name, checkpoint_path1)
        ppo_agent2 = PPO.load(model_name, checkpoint_path2)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward1 = 0
    print_running_reward2 = 0
    print_running_episodes = 0

    log_running_reward1 = 0
    log_running_reward2 = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:

        state, opp_state, _ = env.reset()
        current_ep_reward1 = 0
        current_ep_reward2 = 0

        for t in range(1, max_ep_len+1):
            # select action with policy
            action1 = ppo_agent1.select_action(state)
            action2 = ppo_agent2.select_action(opp_state)
            state, reward1, done, _ = env.step(action1, 1)
            opp_state, reward2, done, _ = env.step(action2, 2)

            # saving reward and is_terminals
            ppo_agent1.buffer.rewards.append(reward1)
            ppo_agent1.buffer.is_terminals.append(done)
            ppo_agent2.buffer.rewards.append(reward2)
            ppo_agent2.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward1 += reward1
            current_ep_reward2 += reward2

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent1.update()
                ppo_agent2.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent1.decay_action_std(action_std_decay_rate, min_action_std)
                ppo_agent2.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward1 = log_running_reward1 / log_running_episodes
                log_avg_reward2 = log_running_reward2 / log_running_episodes
                log_avg_reward1 = round(log_avg_reward1, 4)
                log_avg_reward2 = round(log_avg_reward2, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward1))
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward2))
                log_f.flush()

                log_running_reward1 = 0
                log_running_reward2 = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward1 = print_running_reward1 / print_running_episodes
                print_avg_reward2 = print_running_reward2 / print_running_episodes
                print_avg_reward1 = round(print_avg_reward1, 2)
                print_avg_reward2 = round(print_avg_reward2, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward1))
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward2))

                print_running_reward1 = 0
                print_running_reward2 = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model")
                ppo_agent1.save(checkpoint_path1)
                ppo_agent2.save(checkpoint_path2)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward1 += current_ep_reward1
        print_running_reward2 += current_ep_reward2
        print_running_episodes += 1

        log_running_reward1 += current_ep_reward1
        log_running_reward2 += current_ep_reward2
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")
    print("--------------------------------------------------------------------------------------------")
    print("saving model")
    ppo_agent1.save(checkpoint_path1)
    ppo_agent2.save(checkpoint_path2)
    print("model saved")
    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
    print("--------------------------------------------------------------------------------------------")


if __name__ == '__main__':

    train()
    
    
    
    
    '''
    WAY TO KEEP TRACK OF BETTER POLICY:
    Save models to "curr" directory
    Tests them using test.py and save the better one to "good" dir. The bad to "archive"
    Query "good" for the policy to train using the next set of episodes
    
    '''
    
    
