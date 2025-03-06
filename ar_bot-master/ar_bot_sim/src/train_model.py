from ar_bot_gym.ar_bots_gym import ARBotGymEnv
from PPO_PyTorch.train import train
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name")
parser.add_argument("-c", "--cont", action="store_true")
parser.add_argument("-mt", "--maxTime", type=int)
parser.add_argument("-me", "--maxEp", type=int)

args = parser.parse_args()

if (args.cont and args.name == None):
    sys.stderr.write("ERROR: model name required if continuation is used\n")
    sys.exit(1)

if (args.name != None and args.name[-4:] != ".pth"):
    sys.stderr.write("ERROR: all model names must end with the .pth suffix\n")
    sys.exit(1)

train(ARBotGymEnv, continuation=args.cont, model_name=args.name, max_ep_len=args.maxEp, max_training_timesteps=args.maxTime)

# ssh capstone-2025@mulip-server.eecs.tufts.edu "cd ar_bot-master/ar_bot_sim/src/; python3 train_model.py > /dev/null 2>&1 < /dev/null &"

# scp -P 22 -r capstone-2025@mulip-server.eecs.tufts.edu:/home/capstone-2025/ar_bot-master/ar_bot_sim/src/trained_models/ARBotGymEnv/* .