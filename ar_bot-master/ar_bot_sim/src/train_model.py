from ar_bot_gym.ar_bots_gym import ARBotGymEnv

# from stable_baselines3 import PPO
from PPO_PyTorch.train import train
from PPO_PyTorch.test import test

train(ARBotGymEnv)
# test(ARBotGymEnv)