import numpy as np
import cv2
import time

class Node:
    def __init__(self, position, parent_node, cost_to_come, total_cost):
        self.position = position
        self.parent_node = parent_node
        self.cost_to_come = cost_to_come
        self.total_cost = total_cost

def move_node(distance, angle, current_node, goal):
    angle_radians = np.deg2rad(angle)

    x, y, theta = current_node.position
    x_new = x + distance * np.cos(angle_radians)
    y_new = y + distance * np.sin(angle_radians)
    theta_new = (theta + angle) % 360

    action_cost = distance

    # euclidean distance
    goal_distance = np.linalg.norm(abs(np.array([x_new, y_new]) - np.array(goal[:2])))

    # creating the child node
    child_node = Node((round_x(x_new), round_x(y_new), theta_new),
                      current_node,
                      current_node.cost_to_come + action_cost,
                      current_node.cost_to_come + action_cost + goal_distance)

    return action_cost, goal_distance, child_node


def move_forward(L, current_node, goal):
    return move_node(distance=L, angle=0, current_node=current_node, goal=goal)


def acw_30(L, current_node, goal):
    return move_node(distance=L, angle=30, current_node=current_node, goal=goal)


def cw_30(L, current_node, goal):
    return move_node(distance=L, angle=-30, current_node=current_node, goal=goal)


def acw_60(L, current_node, goal):
    return move_node(distance=L, angle=60, current_node=current_node, goal=goal)


def cw_60(L, current_node, goal):
    return move_node(distance=L, angle=-60, current_node=current_node, goal=goal)

def round_x(x: float) -> int:
    return int(round(x))