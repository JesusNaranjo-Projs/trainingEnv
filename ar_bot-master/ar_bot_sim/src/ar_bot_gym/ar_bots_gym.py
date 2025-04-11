import gym
import pybullet as p
import pybullet_data
import numpy as np
import time
import math
from typing import Optional
from gym import spaces
from pybullet_utils import bullet_client
import sys
import os
import random
from reward_function import compute_team_rewards



class RandomPolicy:
   def __init__(self, action_space):
       self.action_space = action_space
      
   def predict(self, observations):
       return self.action_space.sample(), observations


class ARBotGymEnv(gym.Env):
   """Gym environment for two ARBots in PyBullet. (but one policy is a part of the environment)"""
   metadata = {'render.modes': ['human', 'rgb_array']}
  
   # def __init__(self, gui=True, opponent_policy=None, path = ""):
   def __init__(self, gui=True, path = "", max_timesteps=1500, speed=70):
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
       self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
      
       # Observation space: LiDAR readings + robot positions + ball = 32 ints i think
       self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)

       self.linear_acc_curr = None
       self.angular_acc_curr = None

       self.linear_acc_curr1 = None
       self.angular_acc_curr1 = None

       self.linear_acc_curr2 = None
       self.angular_acc_curr2 = None
       self.prev_ball_pos = None
      
   def _setup_simulation(self):
       """Set up the PyBullet simulation."""
       p.setGravity(0, 0, -10)
       p.setAdditionalSearchPath(pybullet_data.getDataPath())
       p.loadURDF(self.path + "env/maps/arena/arena.urdf")


       # Loads the sphere above the arena drops down in the first step
       sphere_path = self.path + "env/obstacles/sphere_small.urdf"
       y = random.uniform(-0.08, 0.08)
       x = random.uniform(-0.08, 0.08)
       self.ball = p.loadURDF(sphere_path, [x, y, 0.05])
      
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
       self.angular_acc_curr = 0.0
       self.linear_acc_curr = 0.0
       self.linear_acc_curr1 = 0.0
       self.angular_acc_curr1 = 0.0
       self.linear_acc_curr2 = 0.0
       self.angular_acc_curr2 = 0.0
      
      
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

        self.prev_ball_pos = p.getBasePositionAndOrientation(self.ball)[0]

        #[0-1, 0-1]
        angular_acc_delta_norm, linear_acc_delta_norm = action

        #[-10<->10, -0.05<->0.05]
        angular_acc_delta = -10 + angular_acc_delta_norm * (10 - (-10))
        linear_acc_delta = -0.05 + linear_acc_delta_norm * (0.05 - (-0.05))

        self.angular_acc_curr += angular_acc_delta
        self.linear_acc_curr += linear_acc_delta

        if (self.linear_acc_curr * linear_acc_delta) > 0:
            self.linear_acc_curr += linear_acc_delta
        else:
           self.linear_acc_curr = linear_acc_delta

        if (self.angular_acc_curr * angular_acc_delta) > 0:
            self.angular_acc_curr += angular_acc_delta
        else:
            self.angular_acc_curr = angular_acc_delta

        realAct = np.clip([self.linear_acc_curr, self.angular_acc_curr], [-28, -0.4], [28, 0.4])

        #print("realAct", realAct)
        
        if agent_id == 1:
            self._apply_action(self.robot1_id, realAct)
        else:
            self._apply_action(self.robot2_id, realAct)
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
        done, _ = self._is_done(obs)
        reward_main, reward_opponent = self._compute_reward(done)
        
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

        angular_acc_delta1, linear_acc_delta1 = action1
        angular_acc_delta2, linear_acc_delta2 = action2

        if (self.linear_acc_curr1 * linear_acc_delta1) > 0:
            self.linear_acc_curr1 += linear_acc_delta1
        else:
           self.linear_acc_curr1 = linear_acc_delta1

        if (self.angular_acc_curr1 * angular_acc_delta1) > 0:
            self.angular_acc_curr1 += angular_acc_delta1
        else:
            self.angular_acc_curr1 = angular_acc_delta1

        if (self.linear_acc_curr2 * linear_acc_delta2) > 0:      
           self.linear_acc_curr2 += linear_acc_delta2
        else:
           self.linear_acc_curr2 = linear_acc_delta2

        if (self.angular_acc_curr2 * angular_acc_delta2) > 0:
            self.angular_acc_curr2 += angular_acc_delta2
        else:
            self.angular_acc_curr2 = angular_acc_delta2

        realAct1 = np.clip([self.linear_acc_curr1, self.angular_acc_curr1], [-28, -0.4], [28, 0.4])
        realAct2 = np.clip([self.linear_acc_curr2, self.angular_acc_curr2], [-28, -0.4], [28, 0.4])
 
        self._apply_action(self.robot1_id, realAct1)
        self._apply_action(self.robot2_id, realAct2)
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
        done, goal_scorer = self._is_done(obs)
        reward_main, reward_opponent = self._compute_reward()
        info = {}
        
        self.timestep += 1
        if (self.timestep >= self.max_timesteps):
            done = True
        
        return obs, reward_main, reward_opponent, done, goal_scorer
    
   def _apply_action(self, robot_id, action):
       """Apply motion commands to a robot."""
      
       angular, linear = action

       #from urdf file
       r_front = 0.01314
       r_rear = 0.008995
       track_width = 0.048

       v_left = linear - (angular * track_width / 2.0)
       v_right = linear + (angular * track_width / 2.0)

       # Compute angular velocities (rad/s) for each wheel
       left_front_wheel_velocity = v_left / r_front
       left_rear_wheel_velocity  = v_left / r_rear
       right_front_wheel_velocity = v_right / r_front
       right_rear_wheel_velocity  = v_right / r_rear

       p.setJointMotorControl2(robot_id, 5, p.VELOCITY_CONTROL, targetVelocity=left_rear_wheel_velocity, force=1)
       p.setJointMotorControl2(robot_id, 7, p.VELOCITY_CONTROL, targetVelocity=left_front_wheel_velocity, force=1)

       p.setJointMotorControl2(robot_id, 8, p.VELOCITY_CONTROL, targetVelocity=right_front_wheel_velocity, force=1)
       p.setJointMotorControl2(robot_id, 6, p.VELOCITY_CONTROL, targetVelocity=right_rear_wheel_velocity, force=1)

   def _get_observation(self):
       """Get LiDAR readings and robot positions for both robots."""
       lidar1 = self._simulate_lidar(self.robot1_id)
       lidar2 = self._simulate_lidar(self.robot2_id) 
       pos1, orn1 = p.getBasePositionAndOrientation(self.robot1_id)
       pos2, orn2 = p.getBasePositionAndOrientation(self.robot2_id)
       ball_pos, _ =  p.getBasePositionAndOrientation(self.ball)
        
        #gives the lidar position and orientation for each robot plus the balls position
       goal_pos1, _ = p.getBasePositionAndOrientation(self.real_goal_pos1)
       goal_pos2, _ = p.getBasePositionAndOrientation(self.real_goal_pos2)

       dist_ball_goal1 = self.dist(ball_pos, goal_pos1)
       dist_ball_goal2 = self.dist(ball_pos, goal_pos2)

       ball_x, ball_y = ball_pos[:2]
       robot1x, robot1y = pos1[:2]
       robot2x, robot2y = pos2[:2]
       angle_to_ball1 = math.atan2(ball_y - robot1y, ball_x - robot1x)
       angle_to_ball2 = math.atan2(ball_y - robot2y, ball_x - robot2x)
       robot1_yaw = p.getEulerFromQuaternion(orn1)[2]
       robot2_yaw = p.getEulerFromQuaternion(orn2)[2]
       angle_to_turn = angle_to_ball1 - robot1_yaw
       angle_to_turn2 = angle_to_ball2 - robot2_yaw

       angle_to_turn = (angle_to_turn + np.pi) % (2 * np.pi) - np.pi
       angle_to_turn2 = (angle_to_turn2 + np.pi) % (2 * np.pi) - np.pi

       # 0-8 are lidar1, 9-14 are x,y,oobot1, 15-23 are lidar2, 24-29 are x,y,orient of robot2, 30-31 are ball x,y, 32-33 are goal1, 34-35 are goal2
    #    obs = np.hstack((lidar1, pos1[:2], angle_to_turn, lidar2, pos2[:2], angle_to_turn2, ball_pos[:2], dist_ball_goal1, dist_ball_goal2))
    #    print(obs.shape)
       return np.hstack((lidar1, pos1[:2], angle_to_turn, lidar2, pos2[:2], angle_to_turn2, ball_pos[:2], dist_ball_goal1, dist_ball_goal2))
   
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
       return np.clip(distances / lidar_range, 0.0, 1.0)


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
       orient = orient + math.pi


       # Need to account for case where bound wraps around max/min allowed
       diff = abs(ang - orient)


       return orient > math.pi / 6, diff


   # Check if the ball is looking at one of the corners while being close to
   # it.
   def _check_corners(self, obs, robot_id):
       # 11-14 are orient of robot1, 26-29 are orient of robot2
       if robot_id == 1:
           orient = p.getEulerFromQuaternion(obs[11:15])[2]
           robot_pos = obs[9:11]
       elif robot_id == 2:
           orient = p.getEulerFromQuaternion(obs[26:30])[2]
           robot_pos = obs[24:26]
       else:
           return False


       corners = [np.array((0.635, 0.385)), np.array((0.635, -0.385)),
                  np.array((-0.635, 0.385)), np.array((-0.635, -0.385))]
       orient = orient + math.pi


       for corner in corners:
           ang = -math.atan2(robot_pos[1] - corner[1], -(robot_pos[0] - corner[0])) + math.pi
           dist = np.linalg.norm(robot_pos - corner)
          
           if abs(ang - orient) <= math.pi / 6 and dist < 0.5:
               return False
          
       return True
    
    
   def dist(self, p1, p2):
       return np.linalg.norm(np.array(p1) - np.array(p2))
  
   #TODO: needs to be changed with the reward function(prithvi)
   def _compute_reward(self):
        return compute_team_rewards(self)

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

       
       ball_pos, _ =  p.getBasePositionAndOrientation(self.ball)
       goalA_pos, _ = p.getBasePositionAndOrientation(self.real_goal_pos1)
       goalB_pos, _ = p.getBasePositionAndOrientation(self.real_goal_pos2)
       d_ball_goalB = self.dist(ball_pos, goalB_pos)
       d_ball_goalA = self.dist(ball_pos, goalA_pos)

       if d_ball_goalA < 0.075:
           return True, 1
       elif d_ball_goalB < 0.075:
           return True, 2
      
       return False, 0
  
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

