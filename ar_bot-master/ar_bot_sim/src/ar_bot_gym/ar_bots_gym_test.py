import time
import numpy as np
from ar_bots_gym import ARBotGymEnv  # Ensure the environment script is correctly named

# Initialize the environment
env = ARBotGymEnv(gui=True)

# Reset the environment
obs = env.reset()
i = 0
# Run a simple loop to check rendering and movement
for _ in range(200):
    print(i)
    action = np.random.uniform(-1, 1, size=4)  # Random actions for both robots
    obs, reward, done, info = env.step(action)
    
    env.render()  # Render the environment
    i += 1
    time.sleep(6)  # Control the speed of simulation
    
    if done:
        break

# Close the environment
env.close()
