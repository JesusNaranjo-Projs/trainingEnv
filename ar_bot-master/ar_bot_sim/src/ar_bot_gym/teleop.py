import gym
import pybullet as p
import numpy as np
import time
from ar_bots_gym import ARBotGymEnv
import csv
import os
from io import StringIO
import dotenv
import time

# Initialize the environment with GUI enabled
env = ARBotGymEnv(gui=True)

# Define control keys for each robot
key_mapping_robot1 = {
    ord('w'): (0.05, 0),    # Move forward
    ord('s'): (-0.05, 0),   # Move backward
    ord('a'): (0, 10),    # Turn left
    ord('d'): (0, -10)    # Turn right
}


def get_keyboard_action():
    """Reads keyboard input and converts it into actions for both robots."""
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    keys = p.getKeyboardEvents()
    
    # Initialize actions to zero
    action1 = np.array([0, 0], dtype=np.float32)  

    # Process inputs for Robot 1
    for key, (lin, ang) in key_mapping_robot1.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            action1 += np.array([lin, ang], dtype=np.float32)  # Accumulate movements

    action1 = np.where(action1 > 0, 1, np.where(action1 < 0, 0, 0.5))

    return action1 # Combine both robots' actions

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)
ep = int(os.getenv("episode")) + 1

if not os.path.exists("trajectories.csv"):
    with open("trajectories.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Episode","BallX", "BallY", "Observation", "Action1"])
    if ep != 0:
        ep = 0
        dotenv.set_key(dotenv_file, "episode", "0")

obs, _, (x, y) = env.reset()

output = StringIO()
writer = csv.writer(output)
writer.writerow([ep, x, y, obs, "[0. 0.]"])
try:
    while True:
        # Take action first so obs that is stored is state after actions are taken
        action1 = get_keyboard_action()
        obs, reward, done, info = env.step(action1, 1)
        writer.writerow([ep, x, y, obs, action1])

        # Optional: Print reward and observation
        #print(f"Reward: {reward}, Done: {done}")

        if done:
            with open("trajectories.csv", mode="a", newline="") as csv_file:
                csv_file.write(output.getvalue())
            output.close()
            output = StringIO()
            writer = csv.writer(output)
            dotenv.set_key(dotenv_file, "episode", str(ep))

            ep += 1
            print("Episode done")
            obs, _, (x, y) = env.reset()
            writer.writerow([ep, x, y, obs, "[0. 0.]"])
        
        time.sleep(1./60.)  # Maintain a stable refresh rate

except KeyboardInterrupt:
    print("Exiting teleoperation...")
    env.close()