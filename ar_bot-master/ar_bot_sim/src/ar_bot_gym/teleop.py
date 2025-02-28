import gym
import pybullet as p
import numpy as np
import time
from ar_bots_gym import ARBotGymEnv
import csv
import time
#from ar_bot_gym.ar_RL import ARBotGymEnv  # Ensure this points to your environment

# Initialize the environment with GUI enabled
env = ARBotGymEnv(gui=True)

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
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    keys = p.getKeyboardEvents()
    
    # Initialize actions to zero
    action1 = np.array([0, 0], dtype=np.float32)  
    action2 = np.array([0, 0], dtype=np.float32)
    robotOne = True

    # Process inputs for Robot 1
    for key, (lin, ang) in key_mapping_robot1.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            action1 += np.array([lin, ang], dtype=np.float32)  # Accumulate movements

    # Process inputs for Robot 2
    for key, (lin, ang) in key_mapping_robot2.items():
        if key in keys and keys[key] & p.KEY_IS_DOWN:
            robotOne = False
            action2 += np.array([lin, ang], dtype=np.float32)  # Accumulate movements
    
    if robotOne:
        action = action1
        signal = 1 
    else:
        action = action2
        signal = 0

    return action, signal # Combine both robots' actions

    # """Reads keyboard input and converts it into actions for both robots."""
    # p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    # keys = p.getKeyboardEvents()
    # action1 = np.array([0, 0])  # Default: no movement
    # action2 = np.array([0, 0])

    # # Process inputs for Robot 1
    # for key, (lin, ang) in key_mapping_robot1.items():
    #     if key in keys and keys[key] & p.KEY_IS_DOWN:
    #         action1 = np.array([lin, ang])
    #         print("Action 1: ", action1)

    # # Process inputs for Robot 2
    # for key, (lin, ang) in key_mapping_robot2.items():
    #     if key in keys and keys[key] & p.KEY_IS_DOWN:
    #         action2 = np.array([lin, ang])
    #         print("Action 2: ", action2)

    # return np.hstack((action1, action2))  # Combine both robots' actions
csv_file = open("trajectories.csv", "w")

with open("trajectories.csv", "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["Episode", "Observation", "Action", "robot"])

episode = 0
done = 0
try:
    obs = env.reset()
    while done != 5:
        
        action, agent_id = get_keyboard_action()
        #print("Action: ", action)
        with open("trajectories.csv", mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([episode, obs, action, agent_id])

        obs, reward, done, info = env.step(action)

        # Optional: Print reward and observation
        #print(f"Reward: {reward}, Done: {done}")

        if done:
            episode += 1
            done += 1
            print("Episode done")
            #obs = env.reset()
        
        time.sleep(1./60.)  # Maintain a stable refresh rate

except KeyboardInterrupt:
    print("Exiting teleoperation...")
    env.close()