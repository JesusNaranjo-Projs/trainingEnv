import math
import numpy as np
import pybullet as p

def compute_team_rewards(env):
    """
    Compute reward_A and reward_B based on the state of the environment `env`.
    Assumes env has:
    - self.ball, self.robot1_id, self.robot2_id
    - self.real_goal_pos1, self.real_goal_pos2
    - self.prev_ball_pos
    """

    def get_pos_yaw(body_id):
        pos, orn = p.getBasePositionAndOrientation(body_id)
        euler = p.getEulerFromQuaternion(orn)
        return pos, euler[2]

    def dist(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def angle_between(v1, v2):
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        norm1 = math.hypot(*v1)
        norm2 = math.hypot(*v2)
        return math.acos(dot / (norm1 * norm2 + 1e-6))

    def facing_ball_reward(robot_pos, yaw, ball_pos):
        to_ball = (ball_pos[0] - robot_pos[0], ball_pos[1] - robot_pos[1])
        facing_vec = (math.cos(yaw), math.sin(yaw))
        angle = angle_between(facing_vec, to_ball)
        return max(1.0 - angle / math.pi, 0.0)

    def alignment_reward(robot_pos, ball_pos, goal_pos):
        to_goal = (goal_pos[0] - ball_pos[0], goal_pos[1] - ball_pos[1])
        to_robot = (ball_pos[0] - robot_pos[0], ball_pos[1] - robot_pos[1])
        dot = to_goal[0] * to_robot[0] + to_goal[1] * to_robot[1]
        return 1.0 if dot > 0 else -0.5

    def contact_reward(d):
        return 2.0 if d < 0.1 else 0.0

    def approach_reward(robot_pos, yaw, ball_pos):
        to_ball = (ball_pos[0] - robot_pos[0], ball_pos[1] - robot_pos[1])
        facing = (math.cos(yaw), math.sin(yaw))
        angle = angle_between(facing, to_ball)
        if angle < math.pi / 4:
            return 0.2 / (dist(robot_pos, ball_pos) + 1e-5)
        return 0.0

    def ball_movement_reward(prev_ball, curr_ball, goal_pos):
        dx, dy = curr_ball[0] - prev_ball[0], curr_ball[1] - prev_ball[1]
        ball_move = (dx, dy)
        to_goal = (goal_pos[0] - curr_ball[0], goal_pos[1] - curr_ball[1])
        dot = ball_move[0] * to_goal[0] + ball_move[1] * to_goal[1]
        return dot * 10

    # --- Get positions and orientations from obs ---
    # 0-8 are lidar1, 9-11 are x,y,angle, 12-20 are lidar2, 21-23 are x,y,angle of robot2, 24-25 are ball x,y, 26-27 are prev ball x,y, 28-29 are dist to goals, 30-31 is goalpos1, 32-33 is goalpos2
    ball_pos, _ = p.getBasePositionAndOrientation(env.ball)
    goalA_pos, _ = p.getBasePositionAndOrientation(env.real_goal_pos1)
    goalB_pos, _ = p.getBasePositionAndOrientation(env.real_goal_pos2)
    robotA_pos, yawA = get_pos_yaw(env.robot1_id)
    robotB_pos, yawB = get_pos_yaw(env.robot2_id)

    d_A_ball = dist(robotA_pos, ball_pos)
    d_B_ball = dist(robotB_pos, ball_pos)
    d_ball_goalA = dist(ball_pos, goalA_pos)
    d_ball_goalB = dist(ball_pos, goalB_pos)

    reward_A, reward_B = 0.0, 0.0

    # [1] Scoring
    if d_ball_goalB < 0.075:
        reward_A += 100
        reward_B -= 100
    elif d_ball_goalA < 0.075:
        reward_B += 100
        reward_A -= 100

    # [2] Ball movement reward
    if env.prev_ball_pos is not None:
        reward_A += ball_movement_reward(env.prev_ball_pos, ball_pos, goalB_pos)
        reward_B += ball_movement_reward(env.prev_ball_pos, ball_pos, goalA_pos)

    # [3] Contact
    reward_A += contact_reward(d_A_ball)
    reward_B += contact_reward(d_B_ball)

    # # [4] Behind ball alignment
    # reward_A += 1.5 * alignment_reward(robotA_pos, ball_pos, goalB_pos)
    # reward_B += 1.5 * alignment_reward(robotB_pos, ball_pos, goalA_pos)

    # # [5] Facing the ball
    # reward_A += 0.5 * facing_ball_reward(robotA_pos, yawA, ball_pos)
    # reward_B += 0.5 * facing_ball_reward(robotB_pos, yawB, ball_pos)

    # # [6] Approach reward (encouraging getting close *while* facing)
    # reward_A += approach_reward(robotA_pos, yawA, ball_pos)
    # reward_B += approach_reward(robotB_pos, yawB, ball_pos)

    # # [7] Time penalty
    # reward_A -= 0.01
    # reward_B -= 0.01

    return reward_A, reward_B
