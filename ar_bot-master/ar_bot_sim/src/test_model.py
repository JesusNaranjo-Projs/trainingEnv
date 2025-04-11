from ar_bot_gym.ar_bots_gym import ARBotGymEnv
from PPO_PyTorch.test import test
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name")
parser.add_argument("-p", "--path")
parser.add_argument("-r", "--render", action="store_true", default=False)
parser.add_argument("--rand", action="store_true", default=False)

args = parser.parse_args()

if (args.name != None and args.name[-4:] != ".pth"):
    args.name += ".pth"
    # sys.stderr.write("ERROR: all model names must end with the .pth suffix\n")
    # sys.exit(1)

if args.path != None and not os.path.exists(args.path):
    sys.stderr.write("ERROR: given path does not exist\n")
    sys.exit(1)

if args.path != None and args.path[-1] == "/":
    args.path = args.path[:-1]

test(ARBotGymEnv, model_name=args.name, path=args.path, render=args.render,
     rand=args.rand)
