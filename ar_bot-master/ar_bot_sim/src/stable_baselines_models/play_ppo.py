from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from ar_bot_gym import ARBotGym
env = ARBotGym(True)
check_env(env)
model = PPO.load("ppo_arbot")

obs, info = env.reset()
while True:
    action, _ = model.predict(obs)
    observation, reward, terminated, truncated, info = env.step(action)
