import math
import matplotlib.pyplot as plt

import cv2
import numpy as np
import pathlib

show_animation = True

class AStarPlanner:
    def __init__(self, obs_list_x, obs_list_y, resolution, min_safety_dist):
        self.resolution = resolution
        self.min_safety_dist = min_safety_dist
        self.min_x , self.min_y = 0.0, 0.0
        self.max_x, self.max_y = 0.0, 0.0
        self.x_width, self.y_width = 0.0, 0.0
        self.obstacle_map = None
        # 创建2d网格地图
        self.get_obstacle_map(obs_list_x, obs_list_y)
        





def mian():
    start_x = 20.0
    start_y = 40.0
    gaol_x = 140.0
    goal_y = 40.0
    grid_res = 2.0
    min_safety_dist = 1.0


    # 读取地图
    image = cv2.imread(str(pathlib.Path.cwd()) + "/maps/" + "map1.png")