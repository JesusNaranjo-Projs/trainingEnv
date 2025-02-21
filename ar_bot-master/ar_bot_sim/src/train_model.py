from ar_bot_gym.ar_bots_gym import ARBotGymEnv
from stable_baselines_models.train_arbot import TrainARBot

from stable_baselines3 import PPO

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_episodes", type=int)
parser.add_argument("model_name")
args = parser.parse_args()

model = TrainARBot(ARBotGymEnv, PPO)
model_name = args.model_name
num_episodes = args.num_episodes
total_sum_reward_tracker, total_timestep_tracker = model.train(num_episodes,
    "stable_baselines_models/trained_models/" + model_name,
    "stable_baselines_models/training_data/" + model_name + ".npy")