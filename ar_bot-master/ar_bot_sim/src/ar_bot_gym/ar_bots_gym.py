import gym
import pybullet as p
import pybullet_data
import numpy as np
import time
from gym import spaces
from pybullet_utils import bullet_client

class ARBotGymEnv(gym.Env):
    """Gym environment for two ARBots in PyBullet."""
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, gui=True):
        super(ARBotGymEnv, self).__init__()
        self.gui = gui
        self.client = bullet_client.BulletClient(p.GUI if gui else p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self._setup_simulation()
        self.last_touch = -1
        # Action space: Each robot has [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1]), high=np.array([1, 1, 1, 1]), dtype=np.float32)
        
        # Observation space: LiDAR readings + robot positions + ball = 32 ints i think
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)
        
    def _setup_simulation(self):
        """Set up the PyBullet simulation."""
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("env/maps/arena/arena.urdf")

        # Loads the sphere above the arena drops down in the first step
        sphere_path = "env/obstacles/sphere_small.urdf"
        self.ball = p.loadURDF(sphere_path, [0, 0, 0.05])
        
        # Load goals in green in gui
        self.goal_pos1 = np.array([0.0, -0.585])
        self.goal_pos2 = np.array([0.0, 0.585])
        p.loadURDF("env/obstacles/goal.urdf", [self.goal_pos1[1], self.goal_pos1[0], 0])
        p.loadURDF("env/obstacles/goal.urdf", [self.goal_pos2[1], self.goal_pos2[0], 0])
        
        # Load robots on opposite sides
        self.start_pos1 = np.array([-0.30, 0, 0.00])
        self.start_pos2 = np.array([0.30, 0, 0.00])
        initial_orientation1 = p.getQuaternionFromEuler([0, 0, 0])
        initial_orientation2 = p.getQuaternionFromEuler([0, 0, np.pi])
        self.robot1_id = p.loadURDF("agent/cozmo.urdf", self.start_pos1, initial_orientation1)
        self.robot2_id = p.loadURDF("agent/cozmo.urdf", self.start_pos2, initial_orientation2)
        
    def reset(self):
        """Reset the environment."""
        p.resetSimulation()
        self._setup_simulation()
        return self._get_observation()
    
    def render(self, mode='human'):
        """Render the simulation by stepping through PyBullet GUI."""
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    
    
    def step(self, action):
        """Apply actions to both robots and return state, reward, done, and info."""

        #TODO: tune, current duration is equal to 250hz rn
        duration = 250
        for _ in range(duration):
            self._apply_action(self.robot1_id, action[:2])
            self._apply_action(self.robot2_id, action[2:])
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
        reward = self._compute_reward()
        done = self._is_done()
        info = {}
        
        return obs, reward, done, info
    
    def _apply_action(self, robot_id, action):
        """Apply motion commands to a robot."""
        linear, angular = action
        speed = 10
        left_wheel_vel = (linear - angular) * speed
        right_wheel_vel = (linear + angular) * speed
        
        for joint in [5, 7]:  # Left wheels
            p.setJointMotorControl2(robot_id, joint, p.VELOCITY_CONTROL, targetVelocity=left_wheel_vel, force=1000)
        
        for joint in [6, 8]:  # Right wheels
            p.setJointMotorControl2(robot_id, joint, p.VELOCITY_CONTROL, targetVelocity=right_wheel_vel, force=1000)
    
    #TODO: check if this observation space works with stable_baselines3
    def _get_observation(self):
        """Get LiDAR readings and robot positions for both robots."""
        lidar1 = self._simulate_lidar(self.robot1_id)
        lidar2 = self._simulate_lidar(self.robot2_id)
<<<<<<< Updated upstream
=======
<<<<<<< HEAD
        pos1, _ = p.getBasePositionAndOrientation(self.robot1_id)
        pos2, _ = p.getBasePositionAndOrientation(self.robot2_id)

        return np.hstack((lidar1, pos1[:2], lidar2, pos2[:2]))
=======
>>>>>>> Stashed changes
        robPos1, robOrn1 = p.getBasePositionAndOrientation(self.robot1_id)
        robPos2, robOrn2 = p.getBasePositionAndOrientation(self.robot2_id)
        ballPos, _ = p.getBasePositionAndOrientation(self.sphere)
        #[robPos1[:2], robOrn1, lidar1, robPos2[:2], robOrn2, lidar2, ballPos[:2]] <- does not work with stable_baselines3
        return np.hstack((robPos1[:2], robOrn1, lidar1, robPos2[:2], robOrn2, lidar2, ballPos[:2]))
<<<<<<< Updated upstream
=======
>>>>>>> 11f9f936b3103736a1c0609405abc8832b68e2f1
>>>>>>> Stashed changes
    
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
    
    #TODO: needs to be changed with the reward function(prithvi)
    def _compute_reward(self):
        """Compute the reward function."""
        pos1, _ = p.getBasePositionAndOrientation(self.robot1_id)
        pos2, _ = p.getBasePositionAndOrientation(self.robot2_id)
        
        dist1 = np.linalg.norm(np.array(pos1[:2]) - self.goal_pos1)
        dist2 = np.linalg.norm(np.array(pos2[:2]) - self.goal_pos2)
        return -(dist1 + dist2)
    
    """ Checks if the episode is done, specifically if the ball has reached either goal
        Returns a flag if epsiode is done as well as an int for each robot
        1: is the first robot, named self.robot_id1
        2: is the second robot, named self.robot_id2
        -1: is a sentinel that states that something has gone wrong 

        TODO: Need to end epsiode after a specific number of times steps       
    """
    def _is_done(self):
        """Check if the episode is done."""
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball)

        check1 = np.linalg.norm(np.array(ball_pos[:2]) - self.goal_pos1) < 0.1
        check2 = np.linalg.norm(np.array(ball_pos[:2]) - self.goal_pos2) < 0.1

        if(check1 or check2):
            return {True, self.last_touch}
        
        return {False, -1}
    
    def close(self):
        """Close the simulation."""
        p.disconnect()
