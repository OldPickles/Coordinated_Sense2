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

from env_utils import EnvObject, EnvInitData

class Environment:
    def __init__(self, args=None, render_mode='human', n_phone=1, n_uav=1, n_flag=2, n_obstacle=3, n_server=1,
                 agent_vision_length=3, padding=4, width=5, height=5,
                 seed=43, use_init_position=False, use_object_one_hot=True, use_uav_content=True, title="Environment"):
        if args:
            self.args = args
            self.render_mode = args.render_mode
            random.seed(args.seed)
            np.random.seed(args.seed)
            title = args.env_title
            self.agent_vision_length = args.agent_vision_length
            self.padding = max(args.padding,args.agent_vision_length + 1)
            self.width = args.width
            self.height = args.height
            self.pixels = 20

            self.seed = args.seed
            self.use_init_position = args.use_init_position
            if self.use_init_position:
                self.init_position = EnvInitData.init_position
            else:
                self.init_position = None
            self.use_object_one_hot = args.use_object_one_hot
            self.use_uav_content = args.use_uav_content
            self.n_phone = args.n_phone
            self.n_uav = args.n_uav
            self.n_flag = args.n_flag
            self.n_obstacle = args.n_obstacle
            self.n_server = args.n_server
            pass
        else:
            self.render_mode = render_mode
            random.seed(seed)
            np.random.seed(seed)

            self.agent_vision_length = agent_vision_length
            self.padding = max(padding, self.agent_vision_length + 1)
            self.width = width
            self.height = height
            self.pixels = 20

            self.seed = seed
            self.use_init_position = use_init_position
            if self.use_init_position:
                self.init_position = EnvInitData.init_position
            else:
                self.init_position = None
            self.use_object_one_hot = use_object_one_hot
            self.use_uav_content = use_uav_content
            self.n_phone = n_phone
            self.n_uav = n_uav
            self.n_flag = n_flag
            self.n_obstacle = n_obstacle
            self.n_server = n_server

        self.title = title

        self.WIDTH = self.width + self.padding * 2
        self.HEIGHT = self.height + self.padding * 2

        self.n_worker = self.n_phone + self.n_uav
        self.n_object = 5 # 便携式终端:phone， 无人机:uav: ， 任务点:flag，地面数据服务器:server， 障碍物:obstacle
        # 4

        self.avail_actions_shape = (5,)
        self.avail_actions_dim = 5
        self.action_shape = (1,)
        self.action_dim = 1
        self.action_values = [0, 1, 2, 3, 4]
        self.action_value_info = {
            "no_op": 0,
            "up": 1,
            "down": 2,
            "left": 3,
            "right": 4,
        }

        self.worker_one_hot_shape = (self.n_object,)
        self.worker_one_hot_dim = self.n_object

        self.uav_content_shape = (1,)
        self.uav_content_dim = 1

        self.observation_shape = (self.n_object,
                                  1 + self.agent_vision_length*2, 1 + self.agent_vision_length*2)
        # width height 智能体的状态输入时(4*self.width*self.height + self.width*self.height + 2 + 1 )
        # 4*self.width*self.height：任务点分布 ， 数据服务器分布， 无人机分布，智能终端分布
        # self.width*self.height：当前智能体所在位置，只有他自己的位置点。
        # 2：[0,1]表示无人机， [1, 0]表示智能终端。
        # 1：紧迫性
            # 智能终端的数据卸载紧迫性永远为0
            # 无人机的数据卸载紧迫性在0-1之间，例如：无人机硬盘的占用率。

        # 行为空间
        # (self.width*self.height), move_radius*move_radius内为1 否则为0

        # 混合网络输入
        # （4*self.width*self.height）全局状态。

        # rewards
        #

        # 任务点，的数据量。无人机本地
        self.observation_dim = self.observation_shape[0] * self.observation_shape[1] * self.observation_shape[2]
        if self.use_object_one_hot:
            self.observation_dim += self.worker_one_hot_dim
        if self.use_uav_content:
            self.observation_dim += self.uav_content_dim
        self.observation_values = [0, 1]

        self.state_shape = (self.n_object, self.WIDTH, self.HEIGHT)
        self.state_dim = self.state_shape[0] * self.state_shape[1] * self.state_shape[2]
        self.state_values = [0, 1]
        self.state_value_index = {
            "obstacle": 0,
            "phone": 1,
            "uav": 2,
            "flag": 3,
            "server": 4,
        }
        self.state_value_info = {
            "obstacle": 1,
            "phone": 1,
            "uav": 1,
            "flag": 1,
            "server": 1,
        }

        # 走到相应位置获得奖励
        self.reward_info = {
            "flag": 1000,
            "server": 10000,
            "double_reach": 10000,    # 无人机和只能终端均到了同一个小旗子的位置
            "obstacle": -100,
            "road": -1,
        }
        # 奖励两部分计算出来：任务完成方面，数据卸载方面。
        #

        # 5次内，多的卸载数据。
        # 5



        self.reward_shape = (1,)
        self.reward_dim = 1
        self.reward_values = self.reward_info.values()

        self.done_shape = (1,)
        self.done_dim = 1
        self.done_values = [1, 0] # 1: 结束， 0: 继续

        # 环境空间占用情况，初始化全部都是0，表示没有任何东西
        # 共有self.n_object 个矩阵，分别表示对应object的分布
        self.space_occupy = np.full((self.n_object, self.WIDTH, self.HEIGHT),
                                    0, dtype=int)

        # 记录可用空地
        self.avail_positions = list(itertools.product(range(self.padding, self.padding + self.width),
                                                      range(self.padding, self.padding + self.height)))

        # 元素对象:
        self.phones = []
        self.uavs = []
        self.flags = []
        self.obstacles = []
        self.servers = []
        # 元素位置: 服务器模式使用
        self.phone_positions = []
        self.uav_positions = []
        self.flag_positions = []
        self.obstacle_positions = []
        self.server_positions = []

        # 图形化界面对应的元素
        if self.render_mode == 'human':
            self.root = tk.Tk()
            self.root.title(self.title)
            self.canvas = tk.Canvas(self.root, width=self.WIDTH * self.pixels, height=self.HEIGHT * self.pixels)
            self.tk_photo_objects = []    # 用于保存图片对象

            # 初始化元素
            img_flag_path = 'images/flag.png'
            img_phone_path = 'images/agent.png'
            img_obstacle_path = 'images/wall.png'
            img_uav_path = 'images/shovel.png'
            img_server_path = 'images/server.png'
            self.flag_object = ImageTk.PhotoImage(Image.open(img_flag_path).resize((self.pixels, self.pixels)))
            self.phone_object = ImageTk.PhotoImage(Image.open(img_phone_path).resize((self.pixels, self.pixels)))
            self.obstacle_object = ImageTk.PhotoImage(Image.open(img_obstacle_path).resize((self.pixels, self.pixels)))
            self.uav_object = ImageTk.PhotoImage(Image.open(img_uav_path).resize((self.pixels, self.pixels)))
            self.server_object = ImageTk.PhotoImage(Image.open(img_server_path).resize((self.pixels, self.pixels)))
        # 服务器模式
        elif self.render_mode == "None":
            self.root = None
            self.canvas = None
        else:
            raise ValueError("render_mode must be 'human' or 'None'")

        self.set_boundary()
        self.build_environment()

        # 记录最初的占用记录
        self.space_occupy_original = self.space_occupy.copy()

    def build_environment(self):
        # 图形模式下建立网格
        if self.render_mode == 'human':
            for column in range(0, self.WIDTH * self.pixels, self.pixels):
                x0, y0, x1, y1 = column, 0, column, self.HEIGHT * self.pixels
                self.canvas.create_line(x0, y0, x1, y1, fill='grey')

            for row in range(0, self.HEIGHT * self.pixels, self.pixels):
                x0, y0, x1, y1 = 0, row, self.WIDTH * self.pixels, row
                self.canvas.create_line(x0, y0, x1, y1, fill='grey')
            self.canvas.pack()
        # 服务器模式下，什么也不干
        else:
            pass
        self.mode_update()

        # 添加元素
        self.init_elements()

        pass

    def set_boundary(self):
        """
        将边界 padding=2 设置为障碍物
        :return:
        """
        # 填充墙壁
        # 上边界
        walls_position = list(itertools.product(range(self.padding), range(self.WIDTH)))
        # 下边界
        walls_position += list(itertools.product(range(self.HEIGHT-self.padding, self.HEIGHT), range(self.WIDTH)))
        # 左边界
        walls_position += list(itertools.product(range(self.padding, self.HEIGHT - self.padding),
                                                      range(self.padding)))
        # 右边界
        walls_position += list(itertools.product(range(self.padding, self.HEIGHT - self.padding),
                                                      range(self.WIDTH - self.padding, self.WIDTH)))
        for _ in walls_position:
            self.space_occupy[self.state_value_index['obstacle'], _[0], _[1]] = self.state_value_info["obstacle"]
            self.obstacles.append(EnvObject("obstacle", _))
            # self.obstacle_positions.append(_)
            if self.render_mode == 'human':
                tk_photo = self.canvas.create_image(_[0] * self.pixels, _[1] * self.pixels,
                                                    image=self.obstacle_object, anchor=tk.NW)
                self.tk_photo_objects.append(tk_photo)
                self.obstacles[-1].tk_id = tk_photo
                # self.obstacles.append([tk_photo, _])
        self.mode_update()
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
        # 添加元素: 智能终端，无人机，任务点，障碍物，地面数据服务器
        elements = ['phone', 'uav', 'flag', 'obstacle', 'server']
        for element in elements:
            for _ in range(eval(f"self.n_{element}")):
                position = self.get_position(element)
                self.space_occupy[self.state_value_index[element], position[0], position[1]] = self.state_value_info[element]
                eval(f"self.{element}s.append(EnvObject('{element}', position))")
                # eval(f"self.{element}_positions.append(position)")
                if self.render_mode == 'human':
                    tk_photo = self.canvas.create_image(position[0] * self.pixels, position[1] * self.pixels,
                                                        image=eval(f"self.{element}_object"), anchor=tk.NW)
                    self.tk_photo_objects.append(tk_photo)
                    eval(f"self.{element}s[-1].set_tk_id(tk_photo)")
                    # eval(f"self.{element}s.append([tk_photo, position])")
                elif self.render_mode == "None":
                    pass
            self.mode_update()
        pass

    def get_position(self, element):
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
            position = self.init_position[element].pop(0)
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
        elements = ['phone', 'uav', 'flag', 'obstacle', 'server']
        for element in elements:
            # eval(f"self.{element}_positions.clear()")
            # 图形模式
            if self.render_mode == 'human':
                for _ in eval(f"self.{element}s"):
                    self.canvas.delete(_.tk_id)
                eval(f"self.{element}s.clear()")
                self.tk_photo_objects.clear()
            # 服务器模式
            elif self.render_mode == "None":
                eval(f"self.{element}s.clear()")
            else:
                pass


        # 重建元素
        # 按照原有分布重新生成元素：phone，uav，flag，obstacle，server
        self.space_occupy = self.space_occupy_original.copy()
        for (key, value) in self.state_value_index.items():
            for (i, j) in itertools.product(range(self.WIDTH), range(self.HEIGHT)):
                if self.space_occupy[value, i, j] == self.state_value_info[key]:
                    # eval(f"self.{key}_positions.append([i, j])")
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
        :param actions: shape = (n_worker, 1)
            前半部分时智能终端的动作，后半部分是无人机的动作
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
            move_x, move_y = [0, 0]
            if action == self.action_value_info['up']:
                position[1] -= 1
                move_y -= 1
            if action == self.action_value_info['down']:
                position[1] += 1
                move_y += 1
            if action == self.action_value_info['left']:
                position[0] -= 1
                move_x -= 1
            if action == self.action_value_info['right']:
                position[0] += 1
                move_x += 1
            if action == self.action_value_info['no_op']:
                pass

            # 计算reward 与 done
            reward, done, remove_flag =self.get_reward_done(worker, position)

            # worker更新
            worker.set_position(position)
            self.space_occupy[self.state_value_index[worker.name], position[0], position[1]] = \
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

    def get_avail_actions(self,):
        """
        所有worker的可用动作
        :return:
        actions: type = np.array, shape = (n_workers, avail_actions_dim)
        avail_actions_dim = 5 分别表示：no_op, up, down, left, right
            值为0表示不可执行，值为1表示可执行
        """
        actions = []
        workers = self.get_workers()
        for worker in workers:
            position = worker.position
            action = [1,] # 第一个no_op表示原地不动始终可行
            for move_x, move_y in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                new_position = (position[0] + move_x, position[1] + move_y)
                if self.space_occupy[self.state_value_index['obstacle'], new_position[0], new_position[1]] == 0:
                    action.append(1)
                else:
                    action.append(0)
            actions.append(action)
        return np.array(actions).reshape((self.n_worker,) + self.avail_actions_shape)

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
        observations: type = np.array, shape = (n_workers, n_objects, agent_vision_length*2+1, agent_vision_length*2+1)
        """
        observations = []
        for worker in self.get_workers():
            observation = []
            position = worker.position
            up_line = position[1] - self.agent_vision_length
            down_line = position[1] + self.agent_vision_length + 1
            left_line = position[0] - self.agent_vision_length
            right_line = position[0] + self.agent_vision_length + 1
            observation = self.space_occupy[:, up_line:down_line, left_line:right_line].copy()
            observation = observation.flatten().tolist()
            if self.use_object_one_hot:
                observation += worker.one_hot
            if self.use_uav_content:
                observation += [worker.content]
            observation = np.array(observation).flatten()
            observations.append(observation)
        observations = np.array(observations).reshape((self.n_worker, self.observation_dim))
        return observations

    def get_worker_one_hot(self,):
        """
        所有worker的one-hot编码
        :return:
        worker_one_hot: type = np.array, shape = (n_worker, worker_one_hot_dim)
        """
        workers = self.get_workers()
        worker_one_hot = []
        for worker in workers:
            worker_one_hot.append(worker.one_hot)
        worker_one_hot = np.array(worker_one_hot).reshape((self.n_worker, self.worker_one_hot_dim))
        return worker_one_hot

    def actions_sample(self, avail=True):
        """
        随机采样动作
        :param avail: type: bool,
            True: 只返回不撞墙的动作
            False:  随机返回动作
        :return:
             actions: type = np.array, shape = (n_workers, 1)
        """
        if avail:
            actions = []
            avail_actions = self.get_avail_actions()
            for i in range(self.n_worker):
                action = random.choice(np.where(avail_actions[i] == 1)[0])
                actions.append(action)
            actions = np.array(actions).reshape((self.n_worker, 1))
        else:
            actions = np.random.randint(0, self.avail_actions_dim, size=(self.n_worker, 1))
        actions = np.array(actions).reshape((self.n_worker, self.action_dim))
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
            # 如果碰到障碍物
            elif self.space_occupy[self.state_value_index['obstacle'], position[0], position[1]] == \
                    self.state_value_info['obstacle']:
                reward = self.reward_info['obstacle']
                done = True
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
                # 到达障碍物
                elif self.space_occupy[self.state_value_index['obstacle'], position[0], position[1]] ==\
                        self.state_value_info['obstacle']:
                    reward = self.reward_info['obstacle']
                    done = True
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
                # 如果碰到障碍物
                elif self.space_occupy[self.state_value_index['obstacle'], position[0], position[1]] == \
                        self.state_value_info['obstacle']:
                    reward = self.reward_info['obstacle']
                    done = True
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

def test_seed(seeds):
    """
    # 测试不同的seed所产生的环境
    :param seed_dict:
    :return:
    """
    envs = []
    for seed in seeds:
        env = Environment(render_mode="human", seed=seed, title=str(seed))
        envs.append(env)
        env.reset()
        env.render()
    pass
    for env in envs:
        print(env.title)
        env.close()

def test_env(render_mode="human", seed=45):
    """
    # 测试环境
    :return:
    """
    env = Environment(render_mode=render_mode, seed=seed, title=str(seed))
    env.reset()
    env.render()
    for i in range(1000):
        action = env.actions_sample()
        rewards, dones = env.step(action)
        print(rewards.flatten(), dones.flatten())
        env.render()
        if any(dones):
            env.reset()



if __name__ == '__main__':
    test_env(render_mode="None", seed=49)



















