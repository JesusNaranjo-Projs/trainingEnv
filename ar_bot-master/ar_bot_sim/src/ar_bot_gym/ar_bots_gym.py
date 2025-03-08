import gym
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from typing import Optional
from gym import spaces
from pybullet_utils import bullet_client

class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
        
    def predict(self, observations):
        return self.action_space.sample(), observations

class ARBotGymEnv(gym.Env):
    """Gym environment for two ARBots in PyBullet. (but one policy is a part of the environment)"""
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    # def __init__(self, gui=True, opponent_policy=None, path = ""):
    def __init__(self, gui=True, path = "", max_timesteps=2000, speed=200):
        super(ARBotGymEnv, self).__init__()
        self.gui = gui
        self.path = path
        self.max_timesteps = max_timesteps
        self.speed = speed
        self.total_sum_reward_tracker = []
        self.total_timestep_tracker = []
        self.episode_reward_tracker = []
        
        self.client = bullet_client.BulletClient(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._setup_simulation()
        self.last_touch = -1
        # Action space: Each robot has [linear_velocity, angular_velocity] -> now only one robot
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        # Observation space: LiDAR readings + robot positions + ball = 32 ints i think
        # 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)
        
    def _setup_simulation(self):
        """Set up the PyBullet simulation."""
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF(self.path + "env/maps/arena/arena.urdf")

        # Loads the sphere above the arena drops down in the first step
        sphere_path = self.path + "env/obstacles/sphere_small.urdf"
        self.ball = p.loadURDF(sphere_path, [0, 0, 0.05])
        
        # Load goals in green in gui
        self.goal_pos1 = np.array([0.0, -0.585])
        self.goal_pos2 = np.array([0.0, 0.585])
        self.real_goal_pos1 = p.loadURDF(self.path + "env/obstacles/goal.urdf", [self.goal_pos1[1], self.goal_pos1[0], 0])
        self.real_goal_pos2 = p.loadURDF(self.path + "env/obstacles/goal.urdf", [self.goal_pos2[1], self.goal_pos2[0], 0])
        
        # Load robots on opposite sides
        self.start_pos1 = np.array([-0.30, 0, 0.00])
        self.start_pos2 = np.array([0.30, 0, 0.00])
        initial_orientation1 = p.getQuaternionFromEuler([0, 0, 0])
        initial_orientation2 = p.getQuaternionFromEuler([0, 0, np.pi])
        
        # Robot 1 is at the top of the arena
        # Robot 2 is at the bottom of the arena
        self.robot1_id = p.loadURDF(self.path + "agent/cozmo.urdf", self.start_pos1, initial_orientation1)
        self.robot2_id = p.loadURDF(self.path + "agent/cozmo.urdf", self.start_pos2, initial_orientation2)

        self.robot1_dist_to_ball = 0.30
        self.robot2_dist_to_ball = 0.30
        
        self.timestep = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment."""
        p.resetSimulation()
        self._setup_simulation()
        obs = self._get_observation()
        return obs, obs, {}
    
    def render(self, mode='human'):
        """Render the simulation by stepping through PyBullet GUI."""
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
    
    def step(self, action, agent_id):
        """Apply actions to both robots and return state, reward, done, and info."""
        #TODO: tune, current duration is equal to 250hz rn
        if agent_id == 1:
            self._apply_action(self.robot1_id, action)
        else:
            self._apply_action(self.robot2_id, action)
        p.stepSimulation()

        contact_points1 = p.getContactPoints(self.robot1_id, self.ball)
        contact_points2 = p.getContactPoints(self.robot2_id, self.ball)

        if contact_points1:  # If robot1 is in contact with the ball
            self.last_touch = 1
        elif contact_points2:  # If robot2 is in contact with the ball
            self.last_touch = 2
        if self.gui:
            time.sleep(1./240.)
        
        obs = self._get_observation()
        # opp_obs = self._get_opponent_observation()
        reward_main, reward_opponent = self._compute_reward(obs)
        done, _ = self._is_done(obs)
        info = {}
        
        self.timestep += 1
        if (self.timestep >= self.max_timesteps):
            done = True
        
        if agent_id == 1:
            return obs, reward_main, done, info
        else:
            return obs, reward_opponent, done, info
    
    def step_both(self, action1, action2):
        """Apply actions to both robots and return state, reward, done, and info."""
        self._apply_action(self.robot1_id, action1)
        self._apply_action(self.robot2_id, action2)
        p.stepSimulation()

        contact_points1 = p.getContactPoints(self.robot1_id, self.ball)
        contact_points2 = p.getContactPoints(self.robot2_id, self.ball)

        if contact_points1:  # If robot1 is in contact with the ball
            self.last_touch = 1
        elif contact_points2:  # If robot2 is in contact with the ball
            self.last_touch = 2
        if self.gui:
            time.sleep(1./240.)
        
        obs = self._get_observation()
        # opp_obs = self._get_opponent_observation()
        reward_main, reward_opponent = self._compute_reward(obs)
        done, _ = self._is_done(obs)
        info = {}
        
        self.timestep += 1
        if (self.timestep >= self.max_timesteps):
            done = True
        
        return obs, reward_main, reward_opponent, done, info
    
    def _apply_action(self, robot_id, action):
        """Apply motion commands to a robot."""
        linear, angular = action
        speed = self.speed
        left_wheel_vel = (linear - angular) * speed
        right_wheel_vel = (linear + angular) * speed
        
        for joint in [5, 7]:  # Left wheels
            p.setJointMotorControl2(robot_id, joint, p.VELOCITY_CONTROL, targetVelocity=left_wheel_vel, force=1000)
        
        for joint in [6, 8]:  # Right wheels
            p.setJointMotorControl2(robot_id, joint, p.VELOCITY_CONTROL, targetVelocity=right_wheel_vel, force=1000)

    def _get_observation(self):
        """Get LiDAR readings and robot positions for both robots."""
        lidar1 = self._simulate_lidar(self.robot1_id)
        lidar2 = self._simulate_lidar(self.robot2_id)
        pos1, orn1 = p.getBasePositionAndOrientation(self.robot1_id)
        pos2, orn2 = p.getBasePositionAndOrientation(self.robot2_id)
        pos3, _ =  p.getBasePositionAndOrientation(self.ball)
        #gives the lidar position and orientation for each robot plus the balls position
        
        goal_pos1, _ = p.getBasePositionAndOrientation(self.real_goal_pos1)
        goal_pos2, _ = p.getBasePositionAndOrientation(self.real_goal_pos2)

        # 0-8 are lidar1, 9-14 are x,y,orient of robot1, 15-23 are lidar2, 24-29 are x,y,orient of robot2, 30-31 are ball x,y, 32-33 are goal1, 34-35 are goal2
        return np.hstack((lidar1, pos1[:2], orn1, lidar2, pos2[:2], orn2, pos3[:2], goal_pos1[:2], goal_pos2[:2]))
    
    def _get_opponent_observation(self):
        """Get LiDAR readings and robot positions for both robots."""
        lidar1 = self._simulate_lidar(self.robot1_id)
        lidar2 = self._simulate_lidar(self.robot2_id)
        pos1, orn1 = p.getBasePositionAndOrientation(self.robot1_id)
        pos2, orn2 = p.getBasePositionAndOrientation(self.robot2_id)
        pos3, _ =  p.getBasePositionAndOrientation(self.ball)
        #gives the lidar position and orientation for each robot plus the balls position
        ## SWAPPED the positions
        goal_pos1, _ = p.getBasePositionAndOrientation(self.real_goal_pos2)
        goal_pos2, _ = p.getBasePositionAndOrientation(self.real_goal_pos1)
        return np.hstack((lidar1, pos1[:2], orn1, lidar2, pos2[:2], orn2, pos3[:2], goal_pos1[:2], goal_pos2[:2]))

    #TODO: check for accuracy
    def _simulate_lidar(self, robot_id):
        """Simulate LiDAR measurements for a robot."""
        num_rays = 9
        lidar_range = 1
        
        pos, orn = p.getBasePositionAndOrientation(robot_id)
        base_yaw = p.getEulerFromQuaternion(orn)[2]
        
        ray_from = []
        ray_to = []
        
        for ray_angle in np.linspace(-np.pi/2, np.pi/2, num_rays):
            angle = base_yaw + ray_angle
            direction = np.array([np.cos(angle), np.sin(angle), 0])
            ray_from.append(pos)
            ray_to.append(pos + lidar_range * direction)
        
        results = p.rayTestBatch(ray_from, ray_to)
        distances = np.array([res[2] for res in results])
        return distances

    def _check_ball_in_FOV(self, obs, robot_id):
        # 11-14 are orient of robot1, 26-29 are orient of robot2
        ball = obs[30:32]
        if robot_id == 1:
            orient = p.getEulerFromQuaternion(obs[11:15])[2]
            robot_pos = obs[9:11]
        elif robot_id == 2:
            orient = p.getEulerFromQuaternion(obs[26:30])[2]
            robot_pos = obs[24:26]
        else:
            return False

        # Calculate angle to ball (this is the orientation that the robot
        # would need to be facing head on). Change range to 0 - 2pi for all positive
        ang = -math.atan2(robot_pos[1] - ball[1], -(robot_pos[0] - ball[0])) + math.pi
        lower = ang - math.pi / 6
        upper = ang + math.pi / 6
        orient = orient + math.pi

        # Need to account for case where bound wraps around max/min allowed
        if upper > math.pi * 2:
            upper = upper - math.pi * 2
            return orient < upper or orient > lower
            
        if lower < 0:
            lower = math.pi * 2 - lower
            return orient > lower or orient < upper

        return orient > lower and orient < upper
    
    #TODO: needs to be changed with the reward function(prithvi)
    def _compute_reward(self, obs):
        """Compute the reward function."""
        # 0-8 are lidar1, 9-14 are x,y,orient of robot1, 15-23 are lidar2, 24-29 are x,y,orient of robot2, 30-31 are ball x,y, 32-33 are goal1, 34-35 are goal2
        ball = obs[30:32]
        rew1, rew2 = 0, 0
        goal_pos1 = obs[32:34]
        goal_pos2 = obs[34:36]
        dist1 = np.linalg.norm(ball - goal_pos1)
        dist2 = np.linalg.norm(ball - goal_pos2)
        
        robot_pos1 = obs[9:11]
        robot_pos2 = obs[24:26]
        
        dist_to_ball1 = np.linalg.norm(ball - robot_pos1)
        dist_to_ball2 = np.linalg.norm(ball - robot_pos2)

        moving_towards1 = (round(dist_to_ball1, 4) - round(self.robot1_dist_to_ball, 4)) < 0
        moving_towards2 = (round(dist_to_ball2, 4) - round(self.robot2_dist_to_ball, 4)) < 0

        if (dist1 < 0.075):
            rew1 = self.max_timesteps * 5
            rew2 = -self.max_timesteps * 5
        elif (dist2 < 0.075):
            rew1 = -self.max_timesteps * 5
            rew2 = self.max_timesteps * 5
        else:
            if moving_towards1 and self._check_ball_in_FOV(obs, 1):
                # Check for ball in narrow FOV
                rew1 = 1
            else:
                rew1 = -dist1
            if moving_towards2 and self._check_ball_in_FOV(obs, 2):
                # check for ball in narrow FOV
                rew2 = 1
            else:
                rew2 = -dist2
            
            # rew1 = -dist1 * 2 - dist_to_ball1
            # rew2 = -dist2 * 2 - dist_to_ball2

        self.robot1_dist_to_ball = dist_to_ball1
        self.robot2_dist_to_ball = dist_to_ball2

        return rew1, rew2
    
    """ Checks if the episode is done, specifically if the ball has reached either goal
        Returns a flag if epsiode is done as well as an int for each robot
        1: is the first robot, named self.robot_id1
        2: is the second robot, named self.robot_id2
        -1: is a sentinel that states that something has gone wrong 

        TODO: Need to end epsiode after a specific number of times steps       
    """
    def _is_done(self, obs):
        """Check if the episode is done."""
        ball = obs[30:32]

        goal_pos1 = obs[32:34]
        goal_pos2 = obs[34:36]

        check1 = np.linalg.norm(ball - goal_pos1) < 0.075
        check2 = np.linalg.norm(ball - goal_pos2) < 0.075

        if (check1 or check2):
            return True, self.last_touch
        
        return False, -1
    
    def random_opponent(self, observation):
        """A simple random opponent policy."""
        return self.action_space.sample()

    def close(self):
        """Close the simulation."""
        p.disconnect()
        
    def collect_statistics(self) -> None:
        '''
        collect statistics function is used to record total sum and total timesteps per episode
        '''
        self.total_sum_reward_tracker.append(sum(self.episode_reward_tracker))
        self.total_timestep_tracker.append(len(self.episode_reward_tracker))

        self.episode_reward_tracker = []