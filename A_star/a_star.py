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
        # 创建2d网格地图
        self.obstacle_map = None
        self.get_obstacle_map(obs_list_x, obs_list_y)

        # 创建动作模型
        self.motion_model = self.get_motion_model()

    class Node:
        def __init__(self, x_idx, y_idx, cost, parent_idx):
            self.x_idx = x_idx 
            self.y_idx = y_idx
            self.cost = cost
            self.parent_idx = parent_idx

        def __str__(self):
            return str(self.x_idx) + "," + str(self.y_idx) + "," + str(self.cost) + "," + str(self.parent_idx)
        
    def search(self, start_x, start_y, goal_x, goal_y):
        '''
            *self.convert_coord_to_idx(start_x, start_y):

                self.convert_coord_to_idx(start_x, start_y) 是一个方法调用，它将坐标 (start_x, start_y) 转换为索引。
                * 操作符用于将方法返回的元组或列表解包为独立的参数传递给 Node 构造函数。
                start_x 和 start_y 是起始坐标，它们是传递给 convert_coord_to_idx 方法的参数。
        '''
        start_node = self.Node(*self.convert_coord_to_idx(start_x, start_y), 0.0, -1)
        goal_node = self.Node(*self.convert_coord_to_idx(goal_x, goal_y), 0.0, -1)

        # TODO: 创建 开放集合 和 关闭集合
        open_set = []
        closed_set = []


        # astart主循环
        while (len(open_set) > 0):


'''
    # 这段代码定义的八个运动模型向量分别表示以下八个方向：
    # [1, 0]：向右移动一个单位。
    # [0, 1]：向上移动一个单位。
    # [-1, 0]：向左移动一个单位。
    # [0, -1]：向下移动一个单位。
    # [1, 1]：向右上对角线移动一个单位。
    # [-1, 1]：向左上对角线移动一个单位。
    # [-1, -1]：向左下对角线移动一个单位。
    # [1, -1]：向右下对角线移动一个单位。
    # 这些向量表示了在二维平面上的八个基本移动方向，包括四个直行方向（右、上、左、下）和四个对角线方向。
    # 每个方向上的移动距离在代码中已经定义好，例如对角线方向的距离是边长的根号2倍。
'''

def get_motion_model():
    motion_model = [[1, 0, 1],
                    [0, 1, 1],
                    [-1, 0, 1],
                    [0, -1, 1],
                    [-1, -1, np.sqrt(2)],
                    [-1, 1, np.sqrt(2)],
                    [1, -1, np.sqrt(2)],
                    [1, 1, np.sqrt(2)]]
    return motion_model

def preprocess_image(image, threshold):
    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 对灰度图进行二值化处理
    _, binary = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    return binary

def extracct_obstacle_map(binary_img):
    obstacle_x_list = []
    obstacle_y_list = []




def mian():
    start_x = 20.0
    start_y = 40.0
    gaol_x = 140.0
    goal_y = 40.0
    grid_res = 2.0
    min_safety_dist = 1.0


    # 读取地图
    image = cv2.imread(str(pathlib.Path.cwd()) + "/maps/" + "map1.png")
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    binary_image = preprocess_image(image, 127)
