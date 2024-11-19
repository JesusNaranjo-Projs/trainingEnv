#!/usr/bin/env python3

"""
Brennan Miller-Klugman

Based off of
    - https://github.com/erwincoumans/pybullet_robots/blob/master/turtlebot.py
    - https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e
Resources used for lidar: 
    - https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/batchRayCast.py
    - https://github.com/axelbr/racecar_gym/blob/master/racecar_gym/bullet/sensors.py
Resources used for camera:
    - https://www.programcreek.com/python/example/122153/pybullet.computeViewMatrixFromYawPitchRoll

Simulator for AR Bot in PyBullet
"""

import pybullet as p
import os
import time
import numpy as np
from pybullet_utils import bullet_client

class ARBotPybullet:
    def __init__(self, client: int, gui: bool, start_position) -> None:
        """class to spawn in and control arbot

        :param client: physics sim client ID
        """
        self.client = client
        self.gui = gui

	# Change this to the correct path for your system
        urdf_path = "/mnt/c/Users/boyla/Desktop/Classwork/Halligan/trainingEnv/anki_description/urdf/cozmo.urdf"

        print(start_position)

        self.arbot = self.client.loadURDF(
            urdf_path, start_position
        )

        self._hit_color = [1, 0, 0]
        self._miss_color = [0, 1, 0]
        self._ray_ids = []

        self.speed = 20#10
    
    #TODO figure out which motor numbers connect to which wheels try 5-8
    #note: 4 is up and down for the box lifter
    def apply_action(self, action: tuple) -> None:
        """
        Performs action

        :param action: tuple consisting of translation and rotation
        """
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
        
        #
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

    def lidar(self) -> list:
        """simulate lidar measurement
        """

        ray_from = []
        ray_to = []
        num_rays = 9

        lidar_range = 1

        robot_translation, robot_orientation = p.getBasePositionAndOrientation(
            self.arbot
        )

        # Cast rays and get measurements
        for i, ray_angle in enumerate(np.linspace(120, 240, num_rays)):      
            ray_angle = (
                np.radians(ray_angle) + p.getEulerFromQuaternion(robot_orientation)[2]
            )

            ray_direction = np.array([np.cos(ray_angle), np.sin(ray_angle), 0])

            lidar_end_pos = robot_translation + lidar_range * ray_direction

            ray_from.append(robot_translation)
            ray_to.append(lidar_end_pos)

            if (self.gui and len(self._ray_ids) < num_rays):
                self._ray_ids.append(p.addUserDebugLine(ray_from[i], ray_to[i], self._miss_color))

        result = p.rayTestBatch(ray_from, ray_to)

        if self.gui:
            for i in range(num_rays):
                hitObjectUid = result[i][0]

                if (hitObjectUid < 0):
                    p.addUserDebugLine(
                        ray_from[i],
                        ray_to[i],
                        self._miss_color,
                        replaceItemUniqueId=self._ray_ids[i]
                    )
                else:
                    hit_location = result[i][3]
                    p.addUserDebugLine(
                        ray_from[i],
                        hit_location,
                        self._hit_color,
                        replaceItemUniqueId=self._ray_ids[i]
                    )

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


class teleoperate:
    def __init__(self) -> None:
        """helper class to allow teleoperation of the arbot"""
        #self.random_generator = np.random.default_rng()

        self.client = bullet_client.BulletClient(p.GUI)

        plane_path = "ar_bot-master/ar_bot_sim/src/ar_bot_pybullet/env/maps/arena/arena.urdf"
        plane = p.loadURDF(plane_path)
	
	#TODO
	#turn cube into ball
        cube_path = "ar_bot-master/ar_bot_sim/src/ar_bot_pybullet/env/obstacles/sphere_small.urdf"

        #number_of_blocks = 1
        #for obstacle in range(number_of_blocks):
        #    obstacle_x = self.random_generator.uniform(-0.25, 0.25)
        #    obstacle_y = self.random_generator.uniform(-0.4, 0.4)
	
        obstacle_y = 0
        obstacle_x = 0
        obstacle = p.loadURDF(cube_path, [obstacle_y, obstacle_x, 0.05])
	
	#TODO
	#spawn two goals at oppsite ends of the box
	#in the center of each side
        goal_path = "ar_bot-master/ar_bot_sim/src/ar_bot_pybullet/env/obstacles/goal.urdf"
	
	#first goal
        #goal_x = self.random_generator.uniform(-0.35, 0.35)
        goal_x = 0.0
        goal_y = -0.585
        p.loadURDF(goal_path, [goal_y, goal_x, 0])
        
        #second goal
        # goal_x2 = 0.0
        # goal_y2 = 0.585
        # p.loadURDF(goal_path, [goal_y2, goal_x2, 0])
        
        

        goal = (goal_y, goal_x)#, goal_y2, goal_x2)
	
	#start_positions are a array of three values [x,y,z] that describe
	#the start position for each cosmo robot
        start_positions = [
	    [0.4, 0.2 ,0.05],
	    [0.4, -0.2 ,0.05],
	    [-0.4, 0.2 ,0.05],
	    [-0.4, -0.2 ,0.05]
	]
	    
	#spawns 4 cosmo robots	    
        arbot = ARBotPybullet(self.client, True, start_positions[0])
        #arbot1 = ARBotPybullet(self.client, True, start_positions[1])
        #arbot2 = ARBotPybullet(self.client, True, start_positions[2])
        #arbot3 = ARBotPybullet(self.client, True, start_positions[3])


	#TODO
	#fix teleoperating functions because arrow keys do not move
	#the cosmo robots wheels
        p.setRealTimeSimulation(1)
        p.setGravity(0, 0, -10)

        forward = 0
        turn = 0

        while 1:
            p.stepSimulation()
            keys = p.getKeyboardEvents()

            robot_translation, _ = p.getBasePositionAndOrientation(
                arbot.arbot
            )
            sphere_translation, _ = p.getBasePositionAndOrientation(
            	obstacle
            )
            dist_to_goal_y = sphere_translation[0] - goal[0]
            dist_to_goal_x = sphere_translation[1] - goal[1]
            
            # dist_to_y2 = sphere_translation[0] - goal[2]
            # dist_to_x2 = sphere_translation[1] - goal[3]
            
            # second_check = -0.05 < dist_to_y2 < 0.05 and -0.05 < dist_to_x2 < 0.05
            #dist_to_goal_y = robot_translation[0] - goal[0]
            #dist_to_goal_x = robot_translation[1] - goal[1]
            if (-0.05 < dist_to_goal_y < 0.05 and -0.05 < dist_to_goal_x < 0.05):
                print(f"Goal Reached")
                break

            for k, v in keys.items():
                if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    turn = -0.75
                if k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED):
                    turn = 0.0001
                if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    turn = 0.75
                if k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED):
                    turn = 0.0001

                if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    forward = 0.75
                if k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED):
                    forward = 0.0001
                if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_TRIGGERED):
                    forward = -0.75
                if k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED):
                    forward = 0.0001

            arbot.apply_action((forward, turn))
            arbot.lidar()

            time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    teleoperate()
