"""
Date: 2025-4-16: 11:35
Author: <OldPickles>
针对于agent_flag_v5做出以下修改：
注：
- 由于智能体不存在观测长度的约束，故而padding=0，即self.WIDTH=self.width, self.HEIGHT=self.height

1. 删除全局观测的obstacle部分，
    - 全局观测的第一个维度的大小：self.n_object 由5变为4
2. 智能体的输入参数含有多个部分：
    - 全局观测：shape=(self.object, self.WIDTH, self.HEIGHT)，代表着所有object的分布
    - 智能体所处位置：shape=(self.WIDTH, self.HEIGHT)，代表当前智能体的所处的位置，一个WIDTH*HEIGHT的二维矩阵，只有该智能体的位置为1，其余位置为0
    - 智能体类型：shape=(self.n_worker,) n_worker在本环境中值为2，代表着智能终端和感知无人机。
        智能终端：[1, 0]
        感知无人机：[0, 1]
    - 数据卸载紧迫性：shape=(1,)，代表着当前数据卸载的紧迫性，0代表不紧迫，1代表紧迫。
        智能终端的紧迫性始终为0，感知无人机的紧迫性为0-1之间的数，越大意味着紧迫性越强。
3. 行为空间的修改：由前后左右中五个选择修改为可移动范围内的任意位置。
    - 所谓可移动范围：实际意义为当前决策时隙内，worker可移动范围，在本例中，形式上可以是5*5的矩阵，即此5*5的矩阵内，是该worker在本次决策中可以达到的位置。
    - 在内容上定义为一个(self.WIDTH,self.HEIGHT) 可移动位置(move_radius * move_radius)为1， 否则为0.

4. 混合网络的输入：
    - 全局观测：shape=(self.object, self.WIDTH, self.HEIGHT)，代表着所有object的分布

5. rewards的修改：奖励函数不应该再是静态的字典形式，而应该细化。
    - 奖励函数定义：衡量当前时间步内，智能体的行为对于完成总体任务的贡献程度，可以考虑两方面：任务完成数量，以及数据卸载量。总体上来讲，数据卸载量越大
    任务完成数量自然越大，但是可以考虑设置两个奖励驱动：既有任务完成数量的奖励，也有数据卸载量的奖励。这样有助于无人机按照正确的行为顺序去决策。
    - 奖励函数的细化：
        - 到达不同位置的奖励，考虑使用字典形式进行奖励。
    - IDEAS:
        - 奖励函数不仅是内容上是客观的，也应该是形式上是主观的。
        所谓形式上主观意味着可以从全局的角度上，人的主观意志【添加一个智能体，先把他训练成一个统治者，然后可以在off-policy中指导Agent_net的探索模式】控制智能体组织的形式【从区域的角度上来讲，可以是多数智能体集中在某个区域去探索，而部分智能体外围较少任务点的区域去探索】，
"""
"""
为环境添加地面固定数据服务器。
1. 可以自定义工作模式：human or None
2. 可以自定义objects的位置，也可以随机生成。
3.
"""
import itertools

import numpy as np
import random
import time
import tkinter as tk
from PIL import Image, ImageTk

from env_utils_v2 import EnvObject, EnvInitData, EnvFunc

class Environment:
    def __init__(self, args=None, render_mode='human', n_phone=1, n_uav=1, n_flag=2, n_obstacle=3, n_server=1,
                 width=5, height=5, phone_move_radius=1, uav_move_radius=3,
                 seed=43, use_init_position=False, title="Environment"):
        if args:
            self.args = args
            self.render_mode = args.render_mode
            random.seed(args.seed)
            np.random.seed(args.seed)
            title = args.env_title
            self.width = args.width
            self.height = args.height
            EnvObject.workers_move_radius_dict["phone"] = args.phone_move_radius
            EnvObject.workers_move_radius_dict["uav"] = args.uav_move_radius

            self.pixels = 20

            self.seed = args.seed
            self.use_init_position = args.use_init_position
            if self.use_init_position:
                self.init_position = EnvInitData.init_position
            else:
                self.init_position = None
            self.n_phone = args.n_phone
            self.n_uav = args.n_uav
            self.n_flag = args.n_flag
            self.n_server = args.n_server
            pass
        else:
            self.render_mode = render_mode
            random.seed(seed)
            np.random.seed(seed)

            self.width = width
            self.height = height
            EnvObject.workers_move_radius_dict["phone"] = phone_move_radius
            EnvObject.workers_move_radius_dict["uav"] = uav_move_radius

            self.pixels = 20

            self.seed = seed
            self.use_init_position = use_init_position
            if self.use_init_position:
                self.init_position = EnvInitData.init_position
            else:
                self.init_position = None
            self.n_phone = n_phone
            self.n_uav = n_uav
            self.n_flag = n_flag
            self.n_obstacle = n_obstacle
            self.n_server = n_server

        self.title = title


        self.n_worker = self.n_phone + self.n_uav
        self.n_object = 4 # 便携式终端:phone， 无人机:uav: ， 任务点:flag，地面数据服务器:server，

        # avail_action_matrix形式上是一个二维矩阵，代表着哪里可去，哪里不可去，可去为1，不可去为0
        self.avail_actions_matrix_shape = (self.width, self.height)
        self.avail_actions_matrix_dim = EnvFunc.shape2dim(self.avail_actions_matrix_shape)
        self.avail_actions_values = [0, 1]

        # avail_actins形式上是一个由position组成的列表
        # avail_actions_shape = (_, action_shape)
        pass


        # action 形式上是一个长度为2的列表，分别代表将要前往的下一个位置的横纵坐标
        self.action_shape = (2,)
        self.action_dim = EnvFunc.shape2dim(self.action_shape)
        self.action_values = set(range(self.width)).update(range(self.height))

        # worker类型的one-hot编码
        self.worker_one_hot_shape = (self.n_object,)
        self.worker_one_hot_dim = EnvFunc.shape2dim(self.worker_one_hot_shape)

        # 无人机已收集数据大小：【容量：capacity】
        self.uav_content_shape = (1,)
        self.uav_content_dim = EnvFunc.shape2dim(self.uav_content_shape)

        self.observation_one_shape = (self.n_object, self.width, self.height)   # 全局观测
        self.observation_two_shape = (self.width, self.height)  # 智能体所处位置
        self.observation_three_shape = (len(EnvObject.ont_host_base),)     # 智能体类型
        self.observation_four_shape = (1,)      # 数据卸载紧迫性

        self.observation_dim = EnvFunc.shapes2dim(self.observation_one_shape, self.observation_two_shape,
                                                   self.observation_three_shape, self.observation_four_shape)

        # 全局信息
        self.state_shape = (self.n_object, self.width, self.height)
        self.state_dim = EnvFunc.shape2dim(self.state_shape)
        self.state_values = [0, 1]
        self.state_value_index = {
            "phone": 0,
            "uav": 1,
            "flag": 2,
            "server": 3,
        }
        self.state_value_info = {
            "phone": 1,
            "uav": 1,
            "flag": 1,
            "server": 1,
        }

        # 走到相应位置获得奖励
        self.reward_info = {
            "flag": 1000,
            "server": 10000,
            "double_reach": 10000,    # 无人机和智能终端均到了同一个小旗子的位置
            "road": -1,
        }

        # 单个智能体的reward
        self.reward_shape = (1,)
        self.reward_dim = EnvFunc.shape2dim(self.reward_shape)
        self.reward_values = self.reward_info.values()

        # 单个智能体的done
        self.done_shape = (1,)
        self.done_dim = EnvFunc.shape2dim(self.done_shape)
        self.done_values = [1, 0] # 1: 结束， 0: 继续

        # 环境空间占用情况，初始化全部都是0，表示没有任何东西
        # 共有self.n_object 个矩阵，分别表示对应object的分布
        self.space_occupy = np.full((self.n_object, self.width, self.height),
                                    0, dtype=int)

        # 记录可用空地
        self.avail_positions = list(itertools.product(range(self.width), range(self.height)))

        # 元素对象:
        self.phones = []
        self.uavs = []
        self.flags = []
        self.servers = []

        # 图形化界面对应的元素
        if self.render_mode == 'human':
            self.root = tk.Tk()
            self.root.title(self.title)
            self.canvas = tk.Canvas(self.root, width=self.width * self.pixels, height=self.height * self.pixels)
            self.tk_photo_objects = []    # 用于保存图片对象

            # 初始化元素
            img_flag_path = 'images/flag.png'
            img_phone_path = 'images/phone.png'
            img_uav_path = 'images/uav.png'
            img_server_path = 'images/server.png'
            self.flag_object = ImageTk.PhotoImage(Image.open(img_flag_path).resize((self.pixels, self.pixels)))
            self.phone_object = ImageTk.PhotoImage(Image.open(img_phone_path).resize((self.pixels, self.pixels)))
            self.uav_object = ImageTk.PhotoImage(Image.open(img_uav_path).resize((self.pixels, self.pixels)))
            self.server_object = ImageTk.PhotoImage(Image.open(img_server_path).resize((self.pixels, self.pixels)))
        # 服务器模式
        elif self.render_mode == "None":
            self.root = None
            self.canvas = None
        else:
            raise ValueError("render_mode must be 'human' or 'None'")

        self.build_environment()

        # 记录最初的占用记录
        self.space_occupy_original = self.space_occupy.copy()

    def build_environment(self):
        # 图形模式下建立网格
        if self.render_mode == 'human':
            # 竖
            for column in range(0, self.width * self.pixels, self.pixels):
                x0, y0, x1, y1 = column, 0, column, self.height * self.pixels
                self.canvas.create_line(x0, y0, x1, y1, fill='grey')
            # 横
            for row in range(0, self.height * self.pixels, self.pixels):
                x0, y0, x1, y1 = 0, row, self.width * self.pixels, row
                self.canvas.create_line(x0, y0, x1, y1, fill='grey')
            self.canvas.pack()
        # 服务器模式下，什么也不干
        else:
            pass
        self.mode_update()

        # 添加元素
        self.init_elements()

        pass


    def mode_update(self):
        """
        更新模式
        :return:
        """
        if self.render_mode == 'human':
            self.canvas.update()
        elif self.render_mode == "None":
            pass
        else:
            raise ValueError("render_mode must be 'human' or 'None'")
        pass

    def init_elements(self):
        """
        添加元素
        :return:
        """
        # 添加元素: 智能终端，无人机，任务点，地面数据服务器
        element_names = ['phone', 'uav', 'flag', 'server']
        for element_name in element_names:
            for _ in range(eval(f"self.n_{element_name}")):
                position = self.get_position(element_name)
                self.space_occupy[self.state_value_index[element_name], position[0], position[1]] = self.state_value_info[element_name]
                eval(f"self.{element_name}s.append(EnvObject('{element_name}', position))")
                if self.render_mode == 'human':
                    tk_photo = self.canvas.create_image(position[0] * self.pixels, position[1] * self.pixels,
                                                        image=eval(f"self.{element_name}_object"), anchor=tk.NW)
                    self.tk_photo_objects.append(tk_photo)
                    eval(f"self.{element_name}s[-1].set_tk_id(tk_photo)")
                elif self.render_mode == "None":
                    pass
            self.mode_update()
        pass

    def get_position(self, element_name):
        """
        随机生成一个未占用的位置
        :return: list [x, y]
        """
        if self.use_init_position:
            """
            init_position 格式为：
            {
                "phone": [(x1, y1), (x2, y2),...],
                "uav": [(x1, y1), (x2, y2),...],
                "flag": [(x1, y1), (x2, y2),...],
                "obstacle": [(x1, y1), (x2, y2),...],
                "server": [(x1, y1), (x2, y2),...],
            }
            """
            position = self.init_position[element_name].pop(0)
        elif self.init_position is None:
            position = random.choice(self.avail_positions)
            self.avail_positions.remove(position)
        else:
            raise ValueError("init_position must be None or a dict")

        return list(position)

    def render(self):
        """
        图形模式下，刷新界面
        :return:
        """
        if self.render_mode == 'human':
            self.mode_update()
        elif self.render_mode == "None":
            pass
        else:
            raise ValueError("render_mode must be 'human' or 'None'")
        pass

    def reset(self):
        """
        重置环境
        :return:
        """
        # 删除元素
        element_names = ['phone', 'uav', 'flag', 'server']
        for element_name in element_names:
            # eval(f"self.{element}_positions.clear()")
            # 图形模式
            if self.render_mode == 'human':
                for _ in eval(f"self.{element_name}s"):
                    self.canvas.delete(_.tk_id)
                eval(f"self.{element_name}s.clear()")
                self.tk_photo_objects.clear()
            # 服务器模式
            elif self.render_mode == "None":
                eval(f"self.{element_name}s.clear()")
            else:
                pass


        # 重建元素
        # 按照原有分布重新生成元素：phone，uav，flag，obstacle，server
        self.space_occupy = self.space_occupy_original.copy()
        for (key, value) in self.state_value_index.items():
            for (i, j) in itertools.product(range(self.width), range(self.height)):
                if self.space_occupy[value, i, j] == self.state_value_info[key]:
                    eval(f"self.{key}s.append(EnvObject('{key}', [i, j]))")
                    if self.render_mode == 'human':
                        tk_photo = self.canvas.create_image(i * self.pixels, j * self.pixels,
                                                            image=eval(f"self.{key}_object"), anchor=tk.NW)
                        self.tk_photo_objects.append(tk_photo)
                        eval(f"self.{key}s[-1].set_tk_id(tk_photo)")
                    elif self.render_mode == "None":
                        pass
                else:
                    pass
        self.mode_update()

    def step(self, actions):
        """
        执行动作
        :param actions: shape = (self.n_worker,) + self.action_shape
            前半部分时智能终端的动作，后半部分是worker的下一步的位置
        :return:
            dones: bool
            rewards: np.array
                    元素类型：int
                    shape：(n_worker, 1)
                    元素含义：每一个worker获得的奖励
        """
        dones = []
        rewards = []
        workers = self.get_workers()
        for worker_index in range(len(workers)):
            worker = workers[worker_index]
            position = worker.position.copy()
            action = actions[worker_index]

            # 转移位置
            # 删除原有位置的元素
            self.space_occupy[self.state_value_index[worker.name], position[0], position[1]] = 0

            # 新位置
            new_position = list(action.copy())
            move_x = new_position[0] - position[0]
            move_y = new_position[1] - position[1]
            # 计算reward 与 done
            reward, done, remove_flag =self.get_reward_done(worker, new_position)

            # worker更新
            worker.set_position(new_position)
            self.space_occupy[self.state_value_index[worker.name], new_position[0], new_position[1]] = \
                self.state_value_info[worker.name]
            if self.render_mode == 'human':
                self.canvas.move(worker.tk_id,
                                 move_x * self.pixels, move_y * self.pixels)
            elif self.render_mode == "None":
                pass

            rewards.append(reward)
            dones.append(done)

            # 移除元素
            for flag in remove_flag:
                self.space_occupy[self.state_value_index['flag'], flag.position[0], flag.position[1]] = 0
                if self.render_mode == 'human':
                    self.canvas.delete(flag.tk_id)
                self.flags.remove(flag)
            self.mode_update()
        return np.array(rewards).reshape((self.n_worker, 1)), np.array(dones).reshape((self.n_worker, 1))

    def get_avail_actions_matrix(self,):
        """
        所有worker的可用动作
        :return:
        actions: type = np.array, shape = (n_worker,) + self.avail_actions_matrix_shape
            形式上来讲是一个三维矩阵，第一维大小等于self.n_worker, 第二维大小等于self.width, 第三维大小等于self.height
        """
        avail_actions = []
        workers = self.get_workers()
        for worker in workers:
            position = worker.position
            action_space = np.full((self.width, self.height), 0, dtype=int)
            worker_move_radius = worker.get_move_radius()
            left_line = max(position[0] - worker_move_radius, 0)
            right_line = min(position[0] + worker_move_radius + 1, self.width)
            up_line = max(position[1] - worker_move_radius, 0)
            down_line = min(position[1] + worker_move_radius + 1, self.height)
            action_space[left_line:right_line, up_line:down_line] = 1
            avail_actions.append(action_space)
        return np.array(avail_actions).reshape((self.n_worker,) + self.avail_actions_matrix_shape)

    def get_avail_actions(self,):
        """
        所有worker的可用动作
        :return:
        actions: type = list, shape = (n_worker, _,) +self.action_shape
            形式上来讲是一个三维矩阵，第一维大小等于self.n_worker, _ 代表可能的行为个数，第三维大小等于self.action_shape
            而且列表中的位置 是按照get_worker的顺序排列好的
        """
        avail_actions_matrix = self.get_avail_actions_matrix()
        avail_actions = []
        for worker in self.get_workers():
            worker_actions = []
            for i in range(self.width):
                for j in range(self.height):
                    if avail_actions_matrix[self.state_value_index[worker.name], i, j] == 1:
                        worker_actions.append([i, j])
            avail_actions.append(worker_actions)
        return avail_actions

    def get_state(self,):
        """
        环境状态
        :return:
        state: type = np.array, shape = (n_objects, WIDTH, HEIGHT)
        """
        state = self.space_occupy
        state = state.reshape(self.state_shape)
        return state

    def get_observations(self,):
        """
        环境观察
        :return:
        observations: type = np.array, dim = n_worker * observation_dim
        """
        observations = []
        for worker in self.get_workers():
            observation = []
            # 全局观测: type: np.array, shape = (self.n_object, self.width, self.height)
            observation_state = self.get_state()
            observation_state = observation_state.flatten().tolist()
            observation += observation_state

            # 智能体位置: type: np.array, shape = (self.width, self.height)
            observation_location = np.zeros((self.width, self.height))
            observation_location[worker.position[0], worker.position[1]] = 1
            observation_location = observation_location.flatten().tolist()
            observation += observation_location

            # 智能体类型: type: list
            observation_one_hot = worker.one_hot
            observation += observation_one_hot

            # 数据卸载紧迫性:type: int
            observation_agency = Environment.get_urgency(worker)
            observation += [observation_agency]

            # 添加到观察列表
            observations.append(observation)
        observations = np.array(observations).reshape((self.n_worker, self.observation_dim))
        return observations

    def get_workers_one_hot(self,):
        """
        所有worker的one-hot编码
        :return:
        worker_one_hot: type = np.array, shape = (n_worker, worker_one_hot_dim)
        """
        workers = self.get_workers()
        workers_one_hot = []
        for worker in workers:
            workers_one_hot.append(worker.one_hot)
        workers_one_hot = np.array(workers_one_hot).reshape((self.n_worker, self.worker_one_hot_dim))
        return workers_one_hot

    def actions_sample(self):
        """
        随机采样动作
        :param
        :return:
             actions: type = np.array, shape = (n_workers,) + action_shape
        """
        actions = []
        avail_actions = self.get_avail_actions()
        for _ in avail_actions:
            action = random.choice(_)
            actions.append(action)
        actions = np.array(actions).reshape((self.n_worker,) + self.action_shape)
        return actions



    def is_done(self,):
        """
        判断所有任务点的数据是否全都感知，且全部放在了server里面
        :return:
        dones: bool
        True：完成任务
        False：未完成任务
        """
        server_content = 0
        for server in self.servers:
            server_content += server.content
        if server_content == self.n_flag * EnvObject.flag_content:
            return True
        else:
            return False

    def get_workers(self,):
        """
        获取所有worker
        :return:
        workers: list
        """
        return self.phones  + self.uavs

    def get_reward_done(self, worker, position):
        """
        计算奖励与done
        :param worker: type: EnvObject
        :param position: type = list, shape = (2,)
        :return:
        reward: float
            如果是智能终端，则奖励依据所处的位置即可直接给出
            如果是无人机，则依据此时无人机的数据容量。
                容量满了，则只有到达数据服务器才有奖励，否则为 -1
                容量不满，则到达任务点就有奖励，否则为 -1
            如果两者在一块时，在无人机容量满时，依旧不给奖励。如果容量不满，则给奖励。
        done: bool
        remove_flags: type = list,
            找到的需要删除的小旗子
        """
        remove_flags = []
        if worker.name == 'phone':
            # 如果找到小旗子
            if self.space_occupy[self.state_value_index['flag'], position[0], position[1]] == \
                    self.state_value_info['flag']:
                reward = self.reward_info['flag']
                done = False
            # 如果到达空地
            else:
                reward = self.reward_info['road']
                done = True
        elif worker.name == 'uav':
            # 如果容量是满的，则只有到达数据服务器才有奖励
            if worker.content == worker.capacity:
                # 到达数据服务器
                if self.space_occupy[self.state_value_index['server'], position[0], position[1]] == \
                        self.state_value_info['server']:
                    # 为该位置的服务器增加数据容量
                    for server in self.servers:
                        if server.position == position:
                            server.content += worker.content
                            worker.content = 0
                            break
                    reward = self.reward_info['server']
                    done = False
                # 到达空地
                else:
                    reward = self.reward_info['road']
                    done = False
            # 如果容量是不满的，则到达任务点就有奖励
            else:
                # 如果找到小旗子, 则奖励
                if self.space_occupy[self.state_value_index['flag'], position[0], position[1]] == \
                        self.state_value_info['flag']:
                    reward = self.reward_info['flag']
                    done = True
                    # 如果此处有智能设备，则拔掉小旗子，增加uav的数据内容
                    if self.space_occupy[self.state_value_index['phone'], position[0], position[1]] == \
                            self.state_value_info['phone']:
                        worker.content += EnvObject.flag_content
                        reward = self.reward_info["double_reach"]

                        # 存储需要删除的小旗子
                        for flag in self.flags:
                            if flag.position == position:
                                remove_flags.append(flag)
                                break
                # 到达服务器
                elif self.space_occupy[self.state_value_index['server'], position[0], position[1]] == \
                        self.state_value_info['server']:
                    reward = self.reward_info['road']
                    done = False
                # 到达空地
                else:
                    reward = self.reward_info['road']
                    done = False
        else:
            raise ValueError("worker name must be 'phone' or 'uav'")
        return reward, done, remove_flags

    def get_server_content(self,):
        """
        获取全部数据服务器的容量
        :return:
        server_content: type = int
        """
        server_content = 0
        for server in self.servers:
            server_content += server.content
        return server_content

    def close(self,):
        """
        关闭环境
        :return:
        """
        if self.render_mode == 'human':
            self.root.destroy()
        elif self.render_mode == "None":
            pass
        else:
            pass

    @staticmethod
    def get_urgency(worker):
        """
        获取数据卸载紧迫性
        :param worker:
            如果worker.name == 'uav' 才有紧迫性
            否则其他为零
        :return:
        """
        if worker.name == 'uav':
            return worker.content / worker.capacity
        else:
            return 0



if __name__ == '__main__':
    # 测试环境
    env = Environment(render_mode='None')
    while True:
        env.reset()
        for i in range(1000):
            actions = env.actions_sample()
            rewards, dones = env.step(actions)
            states = env.get_state()
            observations = env.get_observations()
            if env.is_done():
                print(f"任务完成 回合数：{i}")
                break
            env.render()






































