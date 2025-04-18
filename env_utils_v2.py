"""
This file contains utility functions for the environment.
1. 环境的元素对象
"""

class EnvObject:
    flag_content = 1  # 表示旗子的数据内容
    uav_capacity = 1  # 表示无人机的数据容量
    # 添加速度属性。
    # 决策间隔时间。
    server_capacity = 100  # 表示服务器的数据容量

    # one-hot编码
    ont_host_base = [0, 0, 0, 0,]
    phone_one_hot = [1, 0, 0, 0,]  # 电话的one-hot编码
    uav_one_hot = [0, 1, 0, 0,]  # 无人机的one-hot编码
    flag_one_hot = [0, 0, 1, 0,]  # 旗子的one-hot编码
    server_one_hot = [0, 0, 0, 1,]  # 服务器的one-hot编码

    # workers可移动半径[名字为半径，其实其实际意义代表着四方空间的范围]
    workers_move_radius_dict = {
        "phone": 2,
        "uav": 3,
        "server": 0,
        "flag": 0,
    }

    def __init__(self, name=None, position=None):
        """
        初始化环境元素对象
        :param name: 元素名称："phone", "uav", "flag", "obstacle", "server"
        :param position: 元素位置
        """
        self.name = name
        if position is None:
            raise ValueError("position cannot be None")
        self.position = position
        self.content = 0  # content: 元素已收集数据大小,只有 flag， uav 和 server 有
        self.capacity = 0    # capacity: 元素数据容量,只有 uav 和 server 有
        self.tk_id = 0  # tk_id: 元素的tkinter id,用于绘制元素，只有human模式下才有
        self.move_radius = EnvObject.workers_move_radius_dict[name] \
            if name in EnvObject.workers_move_radius_dict.keys() else 0

        self.one_hot = EnvObject.ont_host_base
        self.init_one_hot()
        self.init_capacity()
        self.init_content()

    def init_one_hot(self):
        """
        初始化one-hot编码
        :return:
        """
        if self.name == "phone":
            self.one_hot = EnvObject.phone_one_hot
        elif self.name == "uav":
            self.one_hot = EnvObject.uav_one_hot
        elif self.name == "flag":
            self.one_hot = EnvObject.flag_one_hot
        elif self.name == "server":
            self.one_hot = EnvObject.server_one_hot
        else:
            raise ValueError("Invalid object name: {}".format(self.name))
        pass

    def init_capacity(self):
        """
        初始化元素数据容量
        uav 的数据容量为 5， server 的数据容量为 100
        :return:
        """
        if self.name == "uav":
            self.capacity = EnvObject.uav_capacity
        elif self.name == "server":
            self.capacity = EnvObject.server_capacity
        else:
            pass
        pass

    def init_content(self):
        """
        初始化元素数据内容
        flag 的数据内容为 1，其他元素的数据内容为 0
        :return:
        """
        if self.name == "flag":
            self.content = EnvObject.flag_content
        else:
            self.content = 0
        pass

    def set_tk_id(self, tk_id):
        """
        设置元素的tkinter id
        :param tk_id:
        :return:
        """
        self.tk_id = tk_id
        pass

    def set_content(self, content):
        """
        设置元素数据内容
        :param content:
        :return:
        """
        self.content = content
        pass

    def set_position(self, position):
        """
        设置元素位置
        :param position:
        :return:
        """
        self.position = position
        pass

    def set_capacity(self, capacity):
        """
        设置元素数据容量
        :param capacity:
        :return:
        """
        self.capacity = capacity
        pass

    def set_move_radius(self, move_radius):
        """
        设置元素的移动半径
        :param move_radius:
        :return:
        """
        self.move_radius = move_radius

    def get_name(self):
        """
        获取元素名称
        :return:
        """
        return self.name

    def get_position(self):
        """
        获取元素位置
        :return:
        """
        return self.position

    def get_content(self):
        """
        获取元素数据内容
        :return:
        """
        return self.content

    def get_capacity(self):
        """
        获取元素数据容量
        :return:
        """
        return self.capacity

    def get_tk_id(self):
        """
        获取元素的tkinter id
        :return:
        """
        return self.tk_id

    def get_one_hot(self):
        """
        获取元素的one-hot编码
        :return:
        """
        return self.one_hot

    def get_move_radius(self):
        """
        获取元素的移动半径
        :return:
        """
        return self.move_radius


class EnvInitData:
    """
    环境初始化数据
    """

    # 智能终端，无人机，服务器，旗子，障碍物的初始位置
    init_position = {
        "phone": [[0, 0]],
        "uav": [[1, 1]],
        "server": [[2, 2]],
        "flag": [[3, 3]],
        "obstacle": [[4, 4], [1, 4], [2, 3]]
    }

class EnvFunc:
    """
    环境的相关功能函数
    """

    @ staticmethod
    def shape2dim(my_shape):
        """
        将shape转换为dim
        :param my_shape: type: tuple/list
        :return: my_dim: type: int
        """
        if len(my_shape) == 0:
            raise ValueError(f"{my_shape}'s length is 0")
        dim = 1
        for _ in my_shape:
            dim *= _

        return dim

    @ staticmethod
    def shapes2dim(*my_shapes):
        """
        将多个shape转换为dim
        :param my_shapes: type: tuple/list
            此处的my_shapes会将接收到的多个shape变成一个元组
        :return: my_dim: type: int
        """
        dim = 0
        for my_shape in my_shapes:
            dim += EnvFunc.shape2dim(my_shape)
        return dim

if __name__ == '__main__':
    # 测试shapes2dim函数
    my_dim = EnvFunc.shapes2dim((2, 3), (4, 5), (6, 7))
    print(my_dim)