
import gymnasium as gym

import numpy as np

from ar_bot_gym.ar_bot_gym import ARBotGym
from ar_bot_pybullet.ar_bot_pybullet import ARBotPybullet

from stable_baselines3 import PPO


# This is the action space for a simple boxed ppo
actions = gym.spaces.box.Box(
            low=np.array([-0.5, -0.5]),
            high=np.array([0.5, 0.5]),
        )

# Change model name if a different model/path is used
model_name = "stable_baselines_models/trained_models/complex_ppo_boxed"
model = PPO.load(model_name)
random_generator = np.random.default_rng(43)

env = ARBotGym(ARBotPybullet, actions, None, random_generator, True, True, True)
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()