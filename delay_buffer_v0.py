import numpy as np


class DelayBuffer:
    def __init__(self,args=None, batch_size=32, buffer_size=5000, env=None):
        """
        :param env:
        """
        if args is not None:
            self.env = args.env
            self.n_worker = self.env.n_worker
            self.buffer_size = args.buffer_size
            self.memory_size = 0
            self.memory_index = 0
            self.batch_size = args.batch_size
        elif env is not None:
            self.env = env
            self.n_worker = env.n_worker
            self.buffer_size = buffer_size
            self.memory_size = 0
            self.memory_index = 0
            self.batch_size = batch_size
        else:
            raise ValueError("args or env must be provided")
        # 经验池实体
        self.buffer = {
            "last_states": np.zeros(((self.buffer_size,) + self.env.state_shape)),
            "next_states": np.zeros(((self.buffer_size,) + self.env.state_shape)),
            "last_observations": np.zeros(((self.buffer_size, self.n_worker,) + (self.env.observation_dim,))),
            "next_observations": np.zeros(((self.buffer_size, self.n_worker,) + (self.env.observation_dim,))),
            "actions": np.zeros(((self.buffer_size, self.n_worker,) + self.env.action_shape), dtype=np.int32),
            "avail_actions": np.zeros(((self.buffer_size, self.n_worker,) + self.env.avail_actions_shape)),
            "next_avail_actions": np.zeros(((self.buffer_size, self.n_worker,) + self.env.avail_actions_shape)),
            "rewards": np.zeros((self.buffer_size, self.n_worker,) + self.env.reward_shape),
            "dones": np.zeros((self.buffer_size, self.n_worker,) + self.env.done_shape),
            # "worker_one_hots": np.zeros((self.buffer_size, self.n_worker,) + self.env.worker_one_hot_dim),
            "terminate": np.zeros((self.buffer_size, 1))
        }

    def store_transition(self, **kwargs):
        """
        存储经验
        """
        self.memory_size = min(self.memory_size + 1, self.buffer_size)
        for key in self.buffer.keys():
            self.buffer[key][self.memory_index] = kwargs[key]
        self.memory_index = (self.memory_index + 1) % self.buffer_size

    def sample(self, batch_size=None):
        """
        随机抽取batch_size个经验
        """
        batch_size = batch_size if batch_size else self.batch_size
        index = np.random.choice(self.memory_size, size=batch_size, replace=False)
        batch = {}
        for key in self.buffer.keys():
            batch[key] = self.buffer[key][index]
        return batch
