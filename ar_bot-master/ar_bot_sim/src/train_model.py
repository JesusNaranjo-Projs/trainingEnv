from ar_bot_gym.ar_bots_gym import ARBotGymEnv

# from stable_baselines3 import PPO
from PPO_PyTorch.train import train
from PPO_PyTorch.test import test

train(ARBotGymEnv)
# test(ARBotGymEnv)


# ssh capstone-2025@mulip-server.eecs.tufts.edu "cd ar_bot-master/ar_bot_sim/src/; python3 train_model.py > /dev/null 2>&1 < /dev/null &"

# scp -P 22 -r capstone-2025@mulip-server.eecs.tufts.edu:/home/capstone-2025/ar_bot-master/ar_bot_sim/src/trained_models trained_models