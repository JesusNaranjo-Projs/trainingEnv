from stable_baselines3 import PPO
from ar_bot_gym.ar_bots_gym import ARBotGymEnv

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model_name")
parser.add_argument("-o", "--opponent")
args = parser.parse_args()

PATH = "stable_baselines_models/trained_models/"

model = PPO.load(PATH + args.model_name + ".zip")
if args.opponent == None:
    opp = None
else:
    opp = PPO.load(PATH + args.opponent + ".zip")

env = ARBotGymEnv(gui=True, path="ar_bot_gym/", opponent_policy=opp)

obs, info = env.reset()
while True:
    action, _ = model.predict(obs)
    observation, reward, terminated, truncated, info = env.step(action)
