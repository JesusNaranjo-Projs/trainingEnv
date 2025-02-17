import gymnasium as gym

import numpy as np

from ar_bot_gym.ar_bots_gym import ARBotGymEnv
from stable_baselines_models.train_arbot import TrainARBot

from stable_baselines3 import PPO

NUM_EPISODES = 1

model = TrainARBot(ARBotGymEnv, PPO)
model_name = "trash"
total_sum_reward_tracker, total_timestep_tracker = model.train(NUM_EPISODES,
    "stable_baselines_models/trained_models/" + model_name,
    "stable_baselines_models/training_data/" + model_name + ".npy")