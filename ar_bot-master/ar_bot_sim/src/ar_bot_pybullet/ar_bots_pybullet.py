# #!/usr/bin/env python3

# """
# Brennan Miller-Klugman

# Based off of
#     - https://github.com/erwincoumans/pybullet_robots/blob/master/turtlebot.py
#     - https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e
# Resources used for lidar: 
#     - https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/batchRayCast.py
#     - https://github.com/axelbr/racecar_gym/blob/master/racecar_gym/bullet/sensors.py
# Resources used for camera:
#     - https://www.programcreek.com/python/example/122153/pybullet.computeViewMatrixFromYawPitchRoll

# Simulator for AR Bot in PyBullet
# """

import pybullet as p
import os
import time
import numpy as np
from pybullet_utils import bullet_client

import pybullet as p
import os
import time
import numpy as np
from pybullet_utils import bullet_client

class ARBotsPybullet:
    def __init__(self, client, gui, start_pos, initial_yaw=0):
        """Class to spawn in and control ARBot"""
        self.client = client
        self.gui = gui
        urdf_path = "agent/cozmo.urdf"

        initial_yaw = np.radians(initial_yaw)  # Change this value to set the desired orientation
        initial_orientation = p.getQuaternionFromEuler([0, 0, initial_yaw])
        self.arbot = self.client.loadURDF(urdf_path, start_pos, initial_orientation)

        self._hit_color = [1, 0, 0]
        self._miss_color = [0, 1, 0]
        self._ray_ids = []

        self.speed = 10

    def apply_action(self, action):
        """Performs action (translation & rotation)"""
        linear, angular = action

        left_wheel_vel = (linear - angular) * self.speed
        right_wheel_vel = (linear + angular) * self.speed

        """ 
        Joint 0: Name: base_footprint_joint
        Joint 1: Name: base_to_head
        Joint 2: Name: head_camera_joint
        Joint 3: Name: imu_joint
        Joint 4: Name: base_to_lift
        Joint 5: Name: front_left_wheel_joint
        Joint 6: Name: front_right_wheel_joint,           0, Parent Link: front_right_wheel
        Joint 7: Name: rear_left_wheel_joint,           0, Parent Link: rear_left_wheel
        Joint 8: Name: rear_right_wheel_joint,           0, Parent Link: rear_right_wheel
        Joint 9: Name: wheel_left_belt,           4, Parent Link: left_belt
        Joint 10: Name: wheel_right_belt,           4, Parent Link: right_belt
        Joint 11: Name: drop_ir_joint,           4, Parent Link: drop_ir
        """
        #front_left
        self.client.setJointMotorControl2(
            self.arbot,
            5,
            p.VELOCITY_CONTROL,
            targetVelocity=left_wheel_vel,
            force=1000,
        )
        
        #rear_left
        self.client.setJointMotorControl2(
            self.arbot,
            7,
            p.VELOCITY_CONTROL,
            targetVelocity=left_wheel_vel,
            force=1000,
        )

        #front_right
        self.client.setJointMotorControl2(
            self.arbot,
            6,
            p.VELOCITY_CONTROL,
            targetVelocity=right_wheel_vel,
            force=1000,
        )
        
        #rear_right
        self.client.setJointMotorControl2(
            self.arbot,
            8,
            p.VELOCITY_CONTROL,
            targetVelocity=right_wheel_vel,
            force=1000,
        )
        
        #left and right belt
        self.client.setJointMotorControl2(
            self.arbot,
            9,
            p.VELOCITY_CONTROL,
            targetVelocity=left_wheel_vel,
            force=1000,
        )
        self.client.setJointMotorControl2(
            self.arbot,
            10,
            p.VELOCITY_CONTROL,
            targetVelocity=right_wheel_vel,
            force=1000,
        )


    def lidar(self):
        """Simulate LiDAR measurement"""
        ray_from = []
        ray_to = []
        num_rays = 9
        lidar_range = 1

        robot_translation, robot_orientation = p.getBasePositionAndOrientation(self.arbot)

        for i, ray_angle in enumerate(np.linspace(-90, 90, num_rays)):      
            ray_angle = np.radians(ray_angle) + p.getEulerFromQuaternion(robot_orientation)[2]
            ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle), 0])
            lidar_end_pos = robot_translation + lidar_range * ray_direction
            ray_from.append(robot_translation)
            ray_to.append(lidar_end_pos)

            if self.gui and len(self._ray_ids) < num_rays:
                self._ray_ids.append(p.addUserDebugLine(ray_from[i], ray_to[i], self._miss_color))

        result = p.rayTestBatch(ray_from, ray_to)

        if self.gui:
            for i in range(num_rays):
                hitObjectUid = result[i][0]
                color = self._miss_color if hitObjectUid < 0 else self._hit_color
                p.addUserDebugLine(ray_from[i], ray_to[i] if hitObjectUid < 0 else result[i][3], color, replaceItemUniqueId=self._ray_ids[i])

        return np.array(result, dtype=object)[:, 2]
    
    def camera(self):
        """Produces top down camera image of environment
        """

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=50,
            yaw=0,
            pitch=-90,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=1, aspect=float(1920) / 1080, nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = p.getCameraImage(
            width=1920,
            height=1080,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )
        return px


class Teleoperate:
    def __init__(self):
        """Class to allow teleoperation of multiple ARBots"""
        self.client = bullet_client.BulletClient(p.GUI)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)


        # Load environment
        plane_path = "env/maps/arena/arena.urdf"
        p.loadURDF(plane_path)

        #load sphere
        sphere_path = "env/obstacles/sphere_small.urdf"
        ball = p.loadURDF(sphere_path, [0, 0, 0.05])
      
        #goal urdf
        goal_path = "env/obstacles/goal.urdf"

        #first goal
        goal_x = 0.0
        goal_y = -0.585
        p.loadURDF(goal_path, [goal_y, goal_x, 0])
        
        #second goal
        goal_x2 = 0.0
        goal_y2 = 0.585
        p.loadURDF(goal_path, [goal_y2, goal_x2, 0])

        # Spawn two Cozmo robots
        self.arbot1 = ARBotsPybullet(self.client, True, [-0.30, 0, 0.05], 0)
        self.arbot2 = ARBotsPybullet(self.client, True, [0.30, 0, 0.05], 180)

        # Set simulation properties
        p.setRealTimeSimulation(1)
        p.setGravity(0, 0, -10)

        # Movement variables
        self.forward1, self.turn1 = 0, 0
        self.forward2, self.turn2 = 0, 0

        self.run_simulation(ball)



    def run_simulation(self, ball):
        """Main loop for controlling two robots"""
        while True:
            p.stepSimulation()
            keys = p.getKeyboardEvents()

            ###TODO: check if goal is reached by either 
            ###the the first or second robot

            # robot1_translation, _ = p.getBasePositionAndOrientation(
            #     self.arbot1.arbot
            # )

            # robot12_translation, _ = p.getBasePositionAndOrientation(
            #     self.arbot2.arbot
            # )

            # sphere_translation, _ = p.getBasePositionAndOrientation(
            # 	ball
            # )

            for k, v in keys.items():
                # Robot 1 (Arrow Keys)
                if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED): self.turn1 = -0.75
                if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED): self.turn1 = 0.0001
                if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED): self.turn1 = 0.75
                if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED): self.turn1 = 0.0001
                if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED): self.forward1 = 0.75
                if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED): self.forward1 = 0.0001
                if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED): self.forward1 = -0.75
                if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED): self.forward1 = 0.0001

                # Robot 2 (WASD Keys)
                if k == ord('d') and (v & p.KEY_WAS_TRIGGERED): self.turn2 = -0.75
                if k == ord('d') and (v & p.KEY_WAS_RELEASED): self.turn2 = 0.0001
                if k == ord('a') and (v & p.KEY_WAS_TRIGGERED): self.turn2 = 0.75
                if k == ord('a') and (v & p.KEY_WAS_RELEASED): self.turn2 = 0.0001
                if k == ord('w') and (v & p.KEY_WAS_TRIGGERED): self.forward2 = 0.75
                if k == ord('w') and (v & p.KEY_WAS_RELEASED): self.forward2 = 0.0001
                if k == ord('s') and (v & p.KEY_WAS_TRIGGERED): self.forward2 = -0.75
                if k == ord('s') and (v & p.KEY_WAS_RELEASED): self.forward2 = 0.0001

            # Apply actions
            self.arbot1.apply_action((self.forward1, self.turn1))
            self.arbot2.apply_action((self.forward2, self.turn2))

            # LiDAR update
            self.arbot1.lidar()
            self.arbot2.lidar()

            time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    Teleoperate()


