import gymnasium as gym
import pybullet as p
import numpy as np
from typing import Optional
from pybullet_utils import bullet_client

class ARBotGym(gym.Env):
    '''
    Gym environment for ARBot
    '''

    metadata = {"render.modes": ["human"]}

    def __init__(self, agent, actions, discrete_action_mapping, random_generator, obstacle, render = False):
        '''
        Setup Gym environment, start pybullet and call reset

        the provided constructor argument "render" determines wheter pybullet is run headlessly
        '''
    
        self.discrete_action_mapping = discrete_action_mapping
        self.agent = agent
        self.render = render
        self.obstacle = obstacle

        self.action_space = actions

        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1.5, -1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0]), # x, y distance to goal, and Lidar readings between 0 and 1
            high=np.array([1.5, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        )

        self.total_sum_reward_tracker = []
        self.total_timestep_tracker = []

        self.episode_reward_tracker = []

        self.random_generator = random_generator

        self.client = bullet_client.BulletClient(p.GUI if self.render is True else p.DIRECT)

        self.client.setTimeStep(1 / 30)

        self.ar_bot = None
        self.goal = None

        self.count = 0
        self.reset()

    def step(self, action):
        '''
        Take action and return observation

        :param action: action to take
        '''
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = self.discrete_action_mapping[action]
        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            linear, angular = action
            action = (self.discrete_action_mapping[linear], self.discrete_action_mapping[angular])

        self.ar_bot.apply_action(action)

        p.stepSimulation()

        robot_translation, _ = p.getBasePositionAndOrientation(
            self.ar_bot.arbot
        )
        reward = -0.1

        dist_to_goal_y = robot_translation[0] - self.goal[0]
        dist_to_goal_x = robot_translation[1] - self.goal[1]

        complete = False

        lidar = list(self.ar_bot.lidar())

        self.count += 1
        if self.count >= 2000:
            complete = True
            self.count = 0

        # check if goal reached, if so give large reward
        if -0.05 < dist_to_goal_y < 0.05 and -0.05 < dist_to_goal_x < 0.05:
            complete = True
            reward = 1000
            self.count = 0

        obs = [dist_to_goal_y, dist_to_goal_x] + lidar

        self.episode_reward_tracker.append(reward)

        if complete: 
            self.collect_statistics()
    
        return np.array(obs, dtype=np.float32), reward, complete, False, {}


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        '''
        Reset robots posistion and goal posistion randomly
        '''

        p.resetSimulation()
        p.setGravity(0, 0, -10)

        plane_path = "ar_bot_pybullet/env/maps/arena/arena.urdf"
        _ = p.loadURDF(plane_path)

        cube_path = "ar_bot_pybullet/env/obstacles/cube.urdf"
        if self.obstacle:
            for _ in range(self.random_generator.integers(0, 3)):
                obstacle_x = self.random_generator.uniform(-0.25, 0.25)
                obstacle_y = self.random_generator.uniform(-0.4, 0.4)

                obstacle = p.loadURDF(cube_path, [obstacle_y, obstacle_x, 0.05])

        # Spawn random goal
        goal_path = "ar_bot_pybullet/env/obstacles/goal.urdf"

        goal_x = self.random_generator.uniform(-0.335, 0.335)
        goal_y = -0.585
        p.loadURDF(goal_path, [goal_y, goal_x, 0])
        
        # Spawn robot randomly
        self.ar_bot = self.agent(self.client, self.render, self.random_generator)

        self.goal = (goal_y, goal_x)

        robot_translation, _ = p.getBasePositionAndOrientation(
            self.ar_bot.arbot
        )

        dist_to_goal_y = robot_translation[0] - self.goal[0]
        dist_to_goal_x = robot_translation[1] - self.goal[1]

        lidar = list(self.ar_bot.lidar())

        obs = [dist_to_goal_y, dist_to_goal_x] + lidar

        return np.array(obs, dtype=np.float32), {}

    def close(self):
        '''
        Close pybullet sim
        '''

        self.collect_statistics

        self.client.disconnect()

    def collect_statistics(self) -> None:
        '''
        collect statistics function is used to record total sum and total timesteps per episode
        '''
        self.total_sum_reward_tracker.append(sum(self.episode_reward_tracker))
        self.total_timestep_tracker.append(len(self.episode_reward_tracker))

        self.episode_reward_tracker = []