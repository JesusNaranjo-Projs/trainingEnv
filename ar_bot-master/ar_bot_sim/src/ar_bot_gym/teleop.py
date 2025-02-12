import gym
import pybullet as p
import numpy as np
import time
from ar_bots_gym import ARBotGymEnv  # Ensure this points to your environment

# Initialize the environment with GUI enabled
env = ARBotGymEnv(gui=True)
obs = env.reset()

# Define control keys for each robot
key_mapping_robot1 = {
    ord('w'): (1, 0),    # Move forward
    ord('s'): (-1, 0),   # Move backward
    ord('a'): (0, 1),    # Turn left
    ord('d'): (0, -1)    # Turn right
}

key_mapping_robot2 = {
    p.B3G_UP_ARROW: (1, 0),    # Move forward
    p.B3G_DOWN_ARROW: (-1, 0), # Move backward
    p.B3G_LEFT_ARROW: (0, 1),  # Turn left
    p.B3G_RIGHT_ARROW: (0, -1) # Turn right
}

def get_keyboard_action():
    """Reads keyboard input and converts it into actions for both robots."""
    keys = p.getKeyboardEvents()
    action1 = np.array([0, 0])  # Default: no movement
    action2 = np.array([0, 0])

    # Process inputs for Robot 1
    for key, (lin, ang) in key_mapping_robot1.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            action1 = np.array([lin, ang])

    # Process inputs for Robot 2
    for key, (lin, ang) in key_mapping_robot2.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            action2 = np.array([lin, ang])

    return np.hstack((action1, action2))  # Combine both robots' actions

try:
    while True:
        action = get_keyboard_action()
        obs, reward, done, info = env.step(action)

        # Optional: Print reward and observation
        print(f"Reward: {reward}, Done: {done}")

        if done:
            obs = env.reset()
        
        time.sleep(1./60.)  # Maintain a stable refresh rate

except KeyboardInterrupt:
    print("Exiting teleoperation...")
    env.close()