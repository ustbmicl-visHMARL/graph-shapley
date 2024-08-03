import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import gc
import gym
from gym import error, spaces
from gym.utils import seeding
sys.path.append(sys.path[0] + '/../..')
sys.path.append(sys.path[0] + '/..')
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42

try:
    from envs.pyplanner2d import EMExplorer
    from envs.utils import load_config, plot_virtual_map, plot_virtual_map_cov, plot_path
except ImportError as e:
    raise error.DependencyNotInstalled('{}. Build em_exploration and export PYTHONPATH=build_dir'.format(e))


class ExplorationEnv(gym.Env):
    metadata = {'render.modes': ['human', 'state'],
                'render.pause': 0.001}

    def __init__(self,
                 map_size,
                 env_index,
                 test):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config = dir_path + '/exploration_env.ini'
        self._config = load_config(config)#加载环境
        self.env_index = env_index#0 编号
        self.map_size = map_size#40m
        self.dist = 0.0
        self.test = test#false
        self._one_nearest_frontier = False
        self.nearest_frontier_point = 0
        self.seed()#
        self._obs = self.reset()#全部都是0.5
        self._viewer = None
        self.reg_out = self._sim._planner_params.reg_out#false
        self._max_steps = self._sim._environment_params.max_steps#5000.0
        self.max_step = self._max_steps#5000.0

        num_actions = self._sim._planner_params.num_actions#动作数量500
        self._step_length = self._sim._planner_params.max_edge_length#最大步2m
        self._rotation_set = np.arange(0, np.pi * 2, np.pi * 2 / num_actions) - np.pi#旋转360 第一个参数为起点，第二个参数为终点，第三个参数为步长
        self._action_set = [np.array([np.cos(t) * self._step_length, np.sin(t) * self._step_length, t]) for t in self._rotation_set]#500个    [2*cos(角度) 2*sin(角度)  角度]
        self.action_space = spaces.Discrete(n=num_actions)#500个  用于创建离散的非负整数空间
        assert (len(self._action_set) == num_actions)
        self._done = False
        self.loop_clo = False#环
        self._frontier = []
        self._frontier_index = []  # 编号

        self.map_resolution = self._sim._virtual_map_params.resolution#一小格2m*2m;
        rows, cols = self._sim._virtual_map.to_array().shape# 行列 40m 40m
        self.leng_i_map = rows#40
        self.leng_j_map = cols#40
        self._max_sigma = self._sim._virtual_map.get_parameter().sigma0#误差协方差 1m
        min_x, max_x = self._sim._map_params.min_x, self._sim._map_params.max_x#-40 40
        min_y, max_y = self._sim._map_params.min_y, self._sim._map_params.max_y#-40 40
        self._pose = spaces.Box(low=np.array([min_x, min_y, -math.pi]),
                                high=np.array([max_x, max_y, math.pi]), dtype=np.float64)#[-40, -40, -pi] [40, 40, pi]
        self._vm_cov_sigma = spaces.Box(low=0, high=self._max_sigma, dtype=np.float64, shape=(rows, cols))#40*40(0,1)协方差
        self._vm_cov_angle = spaces.Box(low=-math.pi, high=math.pi, dtype=np.float64, shape=(rows, cols))#40*40(-pi,pi)角度协方差
        self._vm_prob = spaces.Box(low=0.0, high=1.0, shape=(rows, cols), dtype=np.float64)#40*40(0,1)概率
        self.observation_space = spaces.Tuple([self._pose,
                                               self._vm_prob,
                                               self._vm_cov_sigma,
                                               self._vm_cov_angle])#观察空间 元组 
        self.ext = 20.0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def _get_obs(self):
        cov_array = self._sim._virtual_map.to_cov_array()#(array([[1., 1., 1., ... 1., 1.]]), array([[1.57079633, ...1.57079633]]))每个array都是40*40的矩阵
        self._obs_show = np.array(#array([-17.81471431,...14105733]) 返回四个array 第一个是三元组 后面三个是40*40的矩阵
            [self._sim.vehicle_position.x, self._sim.vehicle_position.y, self._sim.vehicle_position.theta]), \
                         self._sim._virtual_map.to_array(), cov_array[0], cov_array[1] #初始协方差0.5?
        self._obs = self._sim._virtual_map.to_array()#初始协方差0.5?
        return self._obs

    def _get_utility(self, action=None):
        if action is None:
            distance = 0.0#距离0
        else:
            angle_weight = self._config.getfloat('Planner', 'angle_weight')#角度权重0.4
            distance = math.sqrt(action.x ** 2 + action.y ** 2 + angle_weight * action.theta ** 2)#1.6787  2
        return self._sim.calculate_utility(distance)

    def step(self, action):# env.step(action)，将选择的action输入给env，env 按照这个动作走一步进入下一个状态，所以它的返回值有四个：observation：进入的新状态reward：采取这个行动得到的奖励done：当前游戏是否结束info：其他一些信息，如性能表现，延迟等等，可用于调优
        if self._sim._planner_params.reg_out:  #false
            action = self._action_set[action]
        u1 = self._get_utility()#未传参数 3122.4295 相当于初始化？
        self._sim.simulate([action.x, action.y, action.theta])#[0.000000, 0.000000, -2.654383]
        self.dist = self.dist + math.sqrt(action.x ** 2 + action.y ** 2)#0  2
        u2 = self._get_utility(action)#3130.5713  3124.2037
        return self._get_obs(), self.done(), {}#40*40的初始协方差0.5? 是否探索完成的判断标准 观察到85%

    def plan(self):
        if not self._sim.plan():
            self._done = True
            return []

        actions = []
        for edge in self._sim._planner.iter_solution():
            if self._sim._planner_params.reg_out:
                actions.insert(0, (np.abs(np.asarray(self._rotation_set) - edge.get_odoms()[0].theta)).argmin())
            else:
                actions.insert(0, edge.get_odoms()[0].theta)
        return actions

    def rrt_plan(self, goal_key):
        if not self._sim.rrt_plan(goal_key, self._frontier):
            self._done = True
            return []

        actions = []
        for edge in self._sim._planner.iter_solution():
            actions.insert(0, edge.get_odoms()[0])
        return actions

    def line_plan(self, goal_key, fro=[0, 0]):
        actions = self._sim.line_plan(goal_key, fro)#goal_key=6 fro是边界结点的坐标
        return actions

    def actions_all_goals(self):
        key_size = self._sim._slam.key_size()#关键节点6
        land_size = self.get_landmark_size()#地标个数1
        fro_size = len(self._frontier)#边界节点2
        all_actions = [[]] * (key_size + fro_size)#[[],[],...]   8个
        # actions for frontiers
        for i, vi in enumerate(self._frontier):#边界节点的两个坐标
            all_actions[i + key_size] = self.line_plan(key_size, vi)#[2*sin(角度)  2*cos(角度)  角度]

        return all_actions#去向只有边界节点, 每个动作集n组动作

    def rewards_all_goals(self, all_actions):
        key_size = self._sim._slam.key_size()#关键节点6
        land_size = self.get_landmark_size()#地标个数1
        fro_size = len(self._frontier)#边界节点2
        rewards = [np.nan] * (key_size + fro_size)#not a number表示不是一个数字 [nan, nan, nan, nan, nan, nan, nan, nan]

        # calculating rewards for each actions
        for i, _ in enumerate(self._frontier):
            rewards[i + key_size] = self._sim.simulations_reward(all_actions[i + key_size])#[nan, nan, nan, nan, nan, nan, 0.52, -5.1]
        act_max = np.nanargmax(rewards)#找到奖励最大的动作6
        if self.is_nf(act_max):#是否是最近的边界节点, 是就返回true
            self.loop_clo = False#环 如果是最近的节点就没有成环？
            rewards = np.interp(rewards, (np.nanmin(rewards), np.nanmax(rewards)), (-1.0, 0.0))#一元函数插值, 最小的映射成-1, 最大的映射成0
        else:
            self.loop_clo = True#成环
            rewards = np.interp(rewards, (np.nanmin(rewards), np.nanmax(rewards)), (-1.0, 1.0))#一元函数插值, 最小的映射成-1, 最大的映射成1
        rewards[np.isnan(rewards)] = 0#nan的位置设成0 array([ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.])   array([ 0.,  0.,  0.,  0.,  0.,  0., -1., 1.])
        return rewards

    def status(self):
        return self._sim._virtual_map.explored()

    def done(self):
        return self._done or self._sim.step > self._max_steps or self.status() > 0.85

    def get_landmark_error(self, sigma0=1.0):
        error = 0.0
        for key, predicted in self._sim.map.iter_landmarks():
            landmark = self._sim.environment.get_landmark(key)
            error += np.sqrt((landmark.point.x - predicted.point.x) ** 2 + (landmark.point.y - predicted.point.y) ** 2)
        error += sigma0 * (self._sim.environment.get_landmark_size() - self._sim.map.get_landmark_size())
        return error / self._sim.environment.get_landmark_size()

    def get_dist(self):
        return self.dist

    def get_landmark_size(self):
        return self._sim._slam.map.get_landmark_size()

    def get_key_size(self):
        return self._sim._slam.key_size()

    def print_graph(self):
        self._sim._slam.print_graph()

    def max_uncertainty_of_trajectory(self):
        land_size = self.get_landmark_size()
        self._sim._slam.adjacency_degree_get()
        features = np.array(self._sim._slam.features_out())
        return np.amax(features[land_size:])

    def graph_matrix(self):
        self.frontier()
        trace_map = self._sim._virtual_map.to_cov_trace()#40*40协方差
        key_size = self._sim._slam.key_size()#6
        land_size = self.get_landmark_size()#2
        fro_size = len(self._frontier)#2

        self._sim._slam.adjacency_degree_get()
        adjacency = np.array(self._sim._slam.adjacency_out())#6*6
        features = np.array(self._sim._slam.features_out())#6*1
        adjacency = np.pad(adjacency, ((0, fro_size), (0, fro_size)), 'constant')#补成8*8 后面补0
        features = np.pad(features, ((0, fro_size), (0, 0)), 'constant')#补成8*1 后面补0
        #[-19.908531, -16.090712, 1.028989]
        robot_location = [self._sim.vehicle_position.x, self._sim.vehicle_position.y]#智能体位置[12.83122557167864, -18.982977603309333]
        vehicle_position=[self._sim.vehicle_position.x, self._sim.vehicle_position.y, self._sim.vehicle_position.theta]
        # add frontiers to adjacency matrix  增加矩阵的维度
        frontier=[]
        for i in range(fro_size):
            frontier_point = self._frontier[i]
            for j in range(len(self._frontier_index[i])):
                index_node = self._frontier_index[i][j]
                if index_node == 0:
                    self.nearest_frontier_point = i+key_size
                    dist = self.points2dist(frontier_point, robot_location)
                    adjacency[key_size - 1][i + key_size] = dist
                    adjacency[i + key_size][key_size - 1] = dist
                else:
                    dist = self.points2dist(frontier_point, self._sim._slam.get_key_points(index_node - 1))
                    adjacency[index_node - 1][i + key_size] = dist
                    adjacency[i + key_size][index_node - 1] = dist

        # add frontiers to features matrix col 1: trace of cov  协方差
        for i in range(fro_size):
            indx = self.coor2index(self._frontier[i][0], self._frontier[i][1])#返回坐标[10,28]
            f = trace_map[indx[0]][indx[1]]#trace_map位置坐标40*40协方差矩阵取一个
            features[key_size + i][0] = f

        # add frontiers to features matrix col 2: distance to the robot 各个节点与智能体之间的欧式距离
        features_2 = np.zeros(np.shape(features))
        for i in range(key_size):
            key_point = self._sim._slam.get_key_points(i)
            dist = self.points2dist(key_point, robot_location)
            features_2[i][0] = dist
        for i in range(fro_size):
            frontier_point = self._frontier[i]
            dist = self.points2dist(frontier_point, robot_location)#前沿节点
            features_2[key_size + i][0] = dist
            frontier.append(frontier_point)#添加前沿节点坐标

        # add frontiers to features matrix col 5: direction to the robot 相对方向
        features_5 = np.zeros(np.shape(features))
        root_theta = self._sim.vehicle_position.theta
        for i in range(key_size):
            key_point = self._sim._slam.get_key_points(i)
            tdiff = self.diff_theta(key_point, robot_location, root_theta)
            features_5[i][0] = tdiff
        for i in range(fro_size):
            frontier_point = self._frontier[i]
            tdiff = self.diff_theta(frontier_point, robot_location, root_theta)
            features_5[key_size + i][0] = tdiff

        # add frontiers to features matrix col 3: probability
        features_3 = np.zeros(np.shape(features))
        for i in range(key_size):
            key_point = self._sim._slam.get_key_points(i)
            indx = self.coor2index(key_point[0], key_point[1])
            probobility = self._obs[indx[0]][indx[1]]
            features_3[i][0] = probobility
        for i in range(fro_size):
            frontier_point = self._frontier[i]
            indx = self.coor2index(frontier_point[0], frontier_point[1])
            probobility = self._obs[indx[0]][indx[1]]
            features_3[key_size + i][0] = probobility

        # add frontiers to features matrix clo 4: index of locations
        features_4 = np.zeros(np.shape(features))
        for i in range(key_size - 1):
            features_4[i][0] = -1#其他节点
        features_4[key_size - 1][0] = 0#当前位姿
        for i in range(fro_size):
            features_4[key_size + i][0] = 1#边界节点

        features = np.concatenate((features, features_2, features_5, features_3, features_4), axis=1)

        # create global features
        avg_landmarks_error = np.mean(features[1:land_size + 1][:, 0])#第二行和第三行第一列的平均
        global_features = np.array([avg_landmarks_error])
        return adjacency, features, global_features, fro_size, vehicle_position,frontier

    def is_nf(self, id):#是否是最近的边界节点, 是就返回true
        if self.nearest_frontier_point == id:
            return True
        else:
            return False

    def frontier(self):
        vehicle_location = [self._sim.vehicle_position.x, self._sim.vehicle_position.y]#智能体位置[-19.908531099128506, -16.0907121694454]
        [-19.908531, -16.090712, 1.028989]
        a = self._obs < 0.45#小于0.45返回true   不小于0.45返回false
        free_index_i, free_index_j = np.nonzero(a)#索引值数组一定是2维的tuple,描述了不等于0的值在哪些位置 i代表在哪一行 j代表在哪一列

        all_landmarks = []
        all_frontiers = []
        landmark_keys = range(self.get_landmark_size())#range(0,4)生成一个0

        self._frontier = []
        self._frontier_index = []

        for land_key in landmark_keys:
            points = list(self._sim._slam.get_key_points(land_key))#坐标[-5.315963110697262, -10.952648181243163]
            all_landmarks.append(points)#地标的坐标
        #找到中间的点小于0.45且周围有两个及以上在0.5左右的点作为前沿节点
        for ptr in range(len(free_index_i)):#不等于0的值在哪些位置 i代表在哪一行 j代表在哪一列
            cur_i = free_index_i[ptr]#不等于0的值的位置
            cur_j = free_index_j[ptr]
            count = 0
            cur_i_min = free_index_i[ptr] - 1 if free_index_i[ptr] - 1 >= 0 else 0#上下左右各加一
            cur_i_max = free_index_i[ptr] + 1 if free_index_i[ptr] + 1 < self.leng_i_map else self.leng_i_map - 1
            cur_j_min = free_index_j[ptr] - 1 if free_index_j[ptr] - 1 >= 0 else 0
            cur_j_max = free_index_j[ptr] + 1 if free_index_j[ptr] + 1 < self.leng_j_map else self.leng_j_map - 1

            for ne_i in range(cur_i_min, cur_i_max + 1):#range(9,12) 生成9, 10, 11
                for ne_j in range(cur_j_min, cur_j_max + 1):#range(15, 18) 生成15, 16, 17
                    if 0.49 < self._obs[ne_i][ne_j] < 0.51: #[9,15] [9,16] [9,17] [10,15] [10,16] [10,17] [11,15] [11,16] [11,17] 中心点加上周围八个点
                        count += 1#在0.49到0.51之间就加一

            if count >= 2:
                ind2co = self.index2coor(cur_i, cur_j)#[-7,-19]
                if self._sim._map_params.min_x + self.ext <= ind2co[0] <= self._sim._map_params.max_x - self.ext \
                        and self._sim._map_params.min_y + self.ext <= ind2co[
                    1] <= self._sim._map_params.max_y - self.ext:
                    all_frontiers.append(ind2co)

        cur_fro = all_frontiers[self.nearest_frontier(vehicle_location, all_frontiers)]#在所有的前沿节点中选择一个距离智能体最近的
        self._frontier.append(cur_fro)#前沿节点坐标
        self._frontier_index.append([0])#前沿节点编号

        if not self._one_nearest_frontier:
            for ip, p in enumerate(all_landmarks):#地标的坐标集合
                cur_fro = all_frontiers[self.nearest_frontier(p, all_frontiers)]#在所有的前沿节点中选择一个距离地标最近的
                try:
                    self._frontier_index[self._frontier.index(cur_fro)].append(ip + 1)
                except ValueError:
                    self._frontier.append(cur_fro)
                    self._frontier_index.append([ip + 1])

        not_go = []
        for i, vi in enumerate(self._frontier):#距离智能体最近的点和距离地标最近的点[[17.0, -19.0], [17.0, -17.0]]
            temp_list = []
            for j, vj in enumerate(self._frontier[i:]):
                if vi == vj and i not in not_go:
                    temp_list.append(i + j)
                    if i + j != i:
                        not_go.append(i + j)
            self._frontier_index.append(temp_list)

    def nearest_frontier(self, point, all_frontiers):
        min_dist = float("Inf")#最小距离一个正无穷
        min_index = None
        for index, fro_points in enumerate(all_frontiers):
            dist = self.points2dist(point, fro_points)
            if dist < min_dist:
                min_dist = dist
                min_index = index
        return min_index#返回距离最近的点的编号

    def show_frontier(self, index):
        plt.plot(np.transpose(self._frontier)[0], np.transpose(self._frontier)[1], 'mo')
        plt.plot(np.transpose(self._frontier)[0][index], np.transpose(self._frontier)[1][index], 'ro')

    def index2coor(self, matrix_i, matrix_j):#坐标系
        x = (matrix_j + 0.5) * self.map_resolution + self._sim._map_params.min_x#(16+0.5)*2-40=-7      2是一个小格的长度
        y = (matrix_i + 0.5) * self.map_resolution + self._sim._map_params.min_y#(10+0.5)*2-40=-19
        return [x, y]

    def coor2index(self, x, y):#
        map_j = int(round((x - self._sim._map_params.min_x) / self.map_resolution - 0.5))#(17+40)/2-0.5=28
        map_i = int(round((y - self._sim._map_params.min_y) / self.map_resolution - 0.5))#(-19+40)/2-0.5=10
        return [map_i, map_j]

    def points2dist(self, point1, point2):#两点间距离计算
        dist = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        return dist

    def diff_theta(self, point1, point2, root_theta):
        goal_theta = math.atan2(point1[1] - point2[1], point1[0] - point2[0])
        if goal_theta < 0:
            goal_theta = math.pi * 2 + goal_theta
        if root_theta < 0:
            root_theta = math.pi * 2 + root_theta
        diff = goal_theta - root_theta
        if diff < 0:
            diff = math.pi * 2 + diff
        return diff

    def reset(self):
        self._done = False
        while True:
            # Reset seed in configuration
            if not self.test:#训练时重置随机数种子
                seed1 = self.np_random.randint(0, np.iinfo(np.int32).max, dtype=np.int32)
                seed2 = self.np_random.randint(0, np.iinfo(np.int32).max, dtype=np.int32)
            else:
                seed1 = self.env_index
                seed2 = self.env_index
            landmark_size = int(self.map_size ** 2 * 0.005)#平方#地标个数==8
            self._config.set('Simulator', 'seed', str(seed1))
            self._config.set('Planner', 'seed', str(seed1))
            self._config.set('Simulator', 'lo', str(seed2))
            self._config.set('Environment', 'min_x', str(-self.map_size / 2))
            self._config.set('Environment', 'min_y', str(-self.map_size / 2))
            self._config.set('Environment', 'max_x', str(self.map_size / 2))
            self._config.set('Environment', 'max_y', str(self.map_size / 2))
            self._config.set('Simulator', 'num', str(landmark_size))

            # Initialize new instance and perfrom a 360 degree scan of the surrounding     初始化新实例对周围环境进行360度扫描
            self._sim = EMExplorer(self._config)#期望最大化
            for step in range(4):
                odom = 1, 1, math.pi / 2.0
                u1 = self._get_utility()#3200
                self._sim.simulate(odom)

            if self._sim._slam.map.get_landmark_size() < 1:
                print("regenerate a environment")
                self.env_index = self.env_index + 50
                continue

            # Return initial observation 返回初始的观察
            return self._get_obs()

    def render(self, mode='human', close=False, action_index=-1):
        if close:
            return
        if mode == 'human':
            if self._viewer is None:
                self._sim.plot()
                self._viewer = plt.gcf()
                plt.ion()
                plt.tight_layout()
                plt.xlim((self._sim._map_params.min_x + 14, self._sim._map_params.max_x - 14))
                plt.ylim((self._sim._map_params.min_y + 14, self._sim._map_params.max_y - 14))
                plt.show()
            else:
                self._viewer.clf()
                self._sim.plot()
                # plot_path(self._sim._planner, dubins=False)
                plt.xlim((self._sim._map_params.min_x + 14, self._sim._map_params.max_x - 14))
                plt.ylim((self._sim._map_params.min_y + 14, self._sim._map_params.max_y - 14))
                plt.draw()
            if action_index != -1:
                self.show_frontier(action_index)
            plt.pause(ExplorationEnv.metadata['render.pause'])
        elif mode == 'state':
            # print self._obs
            # assert(len(self._obs_show) == 3)
            print (self._viewer is None)
            if self._viewer is None:
                self._viewer = plt.subplots(1, 3, figsize=(18, 6))
                plot_virtual_map(self._sim._virtual_map, self._sim._map_params,
                                 alpha=1.0, ax=self._viewer[1][0])
                plot_virtual_map_cov(self._obs_show[1:], self._max_sigma,
                                     self._sim._map_params,
                                     alpha=1.0, ax=self._viewer[1][1])
                plt.sca(self._viewer[1][2])
                self._sim.plot()
                plt.ion()
                plt.tight_layout()
                plt.show()
            else:
                self._viewer[1][0].clear()
                plot_virtual_map(self._sim._virtual_map, self._sim._map_params,
                                 alpha=1.0, ax=self._viewer[1][0])
                self._viewer[1][1].clear()
                plot_virtual_map_cov(self._obs_show[1:], self._max_sigma,
                                     self._sim._map_params,
                                     alpha=1.0, ax=self._viewer[1][1])
                self._viewer[1][2].clear()
                plt.sca(self._viewer[1][2])
                self._sim.plot()
                plt.draw()
            plt.pause(ExplorationEnv.metadata['render.pause'])
        else:
            super(ExplorationEnv, self).render(mode=mode)


if __name__ == '__main__':
    import sys

    ExplorationEnv.metadata['render.pause'] = 0.001
    lo_num = 7
    map_size = 40
    total_reward = np.empty([0, 0])
    TEST = False

    mode = 'human'
    env = ExplorationEnv(map_size, lo_num, TEST)
    t = 0
    epoch = 0
    done = False
    flag = False
    actions = []
    env.render(mode=mode)

    for i in range(1000):
        adjacency, featrues, global_features, fro_size = env.graph_matrix()

        key_size = env.get_key_size()
        land_size = env.get_landmark_size()
        all_actions = env.actions_all_goals()
        rewards = env.rewards_all_goals(all_actions)
        # print "rewards: ", rewards, "\n"
        rewards[0:key_size] = np.nan
        act_index = np.nanargmax(rewards)
        max_reward = rewards[act_index]

        actions = all_actions[act_index]
        print("###############################")
        temp_reward = 0
        print("max_reward: ", str(max_reward))

        for a in actions:
            obs, reward, done, _ = env.step(a)
            temp_reward += reward
            env.render(mode=mode, action_index=act_index-key_size)
            print("step: ", str(t), "reward: ", str(reward))

            # print "landmark error: ", env.get_landmark_error()
            t = t + 1
            if done:
                break

        ls = env.get_landmark_size()
        epoch += 1
        adjacency, featrues, global_features, fro_size = env.graph_matrix()
        print('done: ', done, 'explored: ', env.status())

        if done:
            print("done")
            print("total steps: ", t)
            print("epoch is: ", epoch)
            print("error: ", env.get_landmark_error())
            input("Press Enter to continue...")
            del env
            gc.collect()
            env = ExplorationEnv(map_size, lo_num, TEST)
            env.render(mode=mode)
            t = 0
            print("new one")
        flag = False
    plt.pause(1e10)
