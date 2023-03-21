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

# placing obstacles on the map
def obstacle_map(width, height):

    num_rows = height + 1
    num_cols = width + 1

    hexagon_angle = np.deg2rad(30)

    map_grid = np.zeros((num_rows, num_cols, 3), dtype=np.uint8)
    map_grid.fill(0)

    # obstacles vertices
    rect1_coord = np.array([[100, 150], [150, 150], [150, 250], [100, 250]])
    rect2_coord = np.array([[100, 100], [150, 100], [150, 0], [100, 0]])
    triangle_coord = np.array([[460, 25], [510, 125], [460, 225]])
    hexagon_coord = np.array([
        [300, 200],
        [300 + 75 * np.cos(hexagon_angle), 200 - 75 * np.sin(hexagon_angle)],
        [300 + 75 * np.cos(hexagon_angle), 50 + 75 * np.sin(hexagon_angle)],
        [300, 50],
        [300 - 75 * np.cos(hexagon_angle), 50 + 75 * np.sin(hexagon_angle)],
        [300 - 75 * np.cos(hexagon_angle), 200 - 75 * np.sin(hexagon_angle)]
    ]).astype(int)

    # placing obstacles on map_grid
    map_grid = cv2.fillPoly(map_grid, pts=[rect1_coord, rect2_coord, triangle_coord, hexagon_coord], color=(159, 43, 104))
    map_grid = cv2.flip(map_grid, 0)

    return map_grid

def check_obstacles(loc, gap):
    x_max, y_max = 601, 251
    x_min, y_min = 0, 0
    x, y, theta = loc

    hexagon_points = [
        (300, 200 + gap),
        (300 + (75 + gap) * np.cos(np.deg2rad(30)), 125 + (75 + gap) * np.sin(np.deg2rad(30))),
        (300 + (75 + gap) * np.cos(np.deg2rad(30)), 125 - (75 + gap) * np.sin(np.deg2rad(30))),
        (300, 50 - gap),
        (300 - (75 + gap) * np.cos(np.deg2rad(30)), 125 - (75 + gap) * np.sin(np.deg2rad(30))),
        (300 - (75 + gap) * np.cos(np.deg2rad(30)), 125 + (75 + gap) * np.sin(np.deg2rad(30)))
    ]

    hexagon_lines = [
        (hexagon_points[1][1] - hexagon_points[0][1]) / (hexagon_points[1][0] - hexagon_points[0][0]),
        hexagon_points[1][0],
        (hexagon_points[3][1] - hexagon_points[2][1]) / (hexagon_points[3][0] - hexagon_points[2][0]),
        (hexagon_points[4][1] - hexagon_points[3][1]) / (hexagon_points[4][0] - hexagon_points[3][0]),
        hexagon_points[4][0],
        (hexagon_points[5][1] - hexagon_points[0][1]) / (hexagon_points[5][0] - hexagon_points[0][0])
    ]

    triangle_points = [
        (460 - gap // 2, 125 + (102 + 2.25 * gap)),
        (511 + 1.12 * gap, 125),
        (460 - gap // 2, 125 - (102 + 2.25 * gap))
    ]

    triangle_lines = [
        triangle_points[0][0],
        (triangle_points[1][1] - triangle_points[0][1]) / (triangle_points[1][0] - triangle_points[0][0]),
        (triangle_points[2][1] - triangle_points[1][1]) / (triangle_points[2][0] - triangle_points[1][0])
    ]

    in_hexagon = (
        y <= hexagon_lines[0] * (x - hexagon_points[0][0]) + hexagon_points[0][1] and
        x <= hexagon_lines[1] and
        y >= hexagon_lines[2] * (x - hexagon_points[3][0]) + hexagon_points[3][1] and
        y >= hexagon_lines[3] * (x - hexagon_points[4][0]) + hexagon_points[4][1] and
        x >= hexagon_lines[4] and
        y <= hexagon_lines[5] * (x - hexagon_points[5][0]) + hexagon_points[5][1]
    )

    in_triangle = (
        x >= triangle_lines[0] and
        y <= triangle_lines[1] * (x - triangle_points[0][0]) + triangle_points[0][1] and
        y >= triangle_lines[2] * (x - triangle_points[1][0]) + triangle_points[1][1]
    )

    in_boundary = (
        x < x_min + gap or
        y < y_min + gap or
        x >= x_max - gap or
        y >= y_max - gap
    )

    in_rectangles = (
        (x <= 150 + gap) and (x >= 100 - gap) and (y <= 100 + gap) or
        (x <= 150 + gap) and (x >= 100 - gap) and (y >= 150 - gap)
    )

    return not (in_hexagon or in_triangle or in_boundary or in_rectangles)

def get_childnode(node, step_size, goal, clearance):
    actions = [(move_forward, 0), (acw_60, 30), (acw_60, 60), (cw_30, -30), (cw_60, -60)]
    children = []
    for action, angle in actions:
        actionCost, cgoal, child = action(step_size, node, goal)
        if check_obstacles(child.position, clearance):
            children.append((actionCost, cgoal, child))
        else:
            del child
    return children

def optimal_path(current):
    path = [current.position]
    while current.parent_node is not None:
        current = current.parent_node
        path.append(current.position)
    return path[::-1]

def astar(start, goal, step_size, clearance):
    start_time = time.time()
    if not all(map(lambda pos: check_obstacles(pos, clearance), [start, goal])):
        print("One of the positions is in the obstacle grid")
        return False

    grid = obstacle_map(600, 250)
    tc_g = np.linalg.norm(np.subtract(start[:2], goal[:2]))
    open_dict, closed_dict, viz = {}, {}, []
    initial_node = Node(start, None, 0, tc_g)
    open_dict[initial_node.position] = initial_node

    while open_dict:
        current = min(open_dict.values(), key=lambda x: x.total_cost)
        if np.linalg.norm(np.subtract(current.position[:2], goal[:2])) <= step_size * 0.5:
            pathTaken = optimal_path(current)
            print('Time taken for A* algorithm:', time.time() - start_time)
            return pathTaken, viz
        else:
            childList = get_childnode(current, step_size, goal, clearance)
            for _, _, child_created in childList:
                if child_created.position in closed_dict:
                    continue
                actionCost = child_created.cost_to_come - current.cost_to_come
                if child_created.position not in open_dict:
                    open_dict[child_created.position] = child_created
                    viz.append(child_created)
                elif actionCost < 0:
                    open_dict[child_created.position].parent_node = current
                    open_dict[child_created.position].cost_to_come = current.cost_to_come + actionCost
        closed_dict[current.position] = current
        open_dict.pop(current.position)
    print("Path not found")
    return False
