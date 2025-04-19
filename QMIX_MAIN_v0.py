from agent_flag_v5 import Environment
from delay_buffer_v0 import DelayBuffer
from qmix_algo import QMixAlgo

from env_utils import EnvInitData

import numpy as np

import matplotlib.pyplot as plt
import random
import time
import os
import argparse

class Runner_QMix:
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self.args.env = env
        self.max_episodes = args.max_episodes
        self.max_steps = args.max_steps
        self.total_episodes = 0
        self.evaluate_freq = args.evaluate_freq

        self.epsilon = args.epsilon
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay_function = args.epsilon_decay_function

        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size

        self.lr = args.lr
        self.use_lr_decay = args.use_lr_decay
        self.gamma = args.gamma
        self.target_update_freq = args.target_update_freq

        self.seed = args.seed

        time_str = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
        self.model_save_path_base = f"{time_str}-max_train_steps-{self.max_episodes}-epsilon_min-{self.epsilon_min}-" + \
                                    f"lr-{self.lr}-" + \
                                    f"target_update_freq-{self.target_update_freq}-qmix_seed-{self.seed}" + \
                                    f"-env_seed-{env.seed}-decay_lr-{self.use_lr_decay}"
        self.base_save_path = args.base_save_path
        self.train_info_file = "train_info.txt"

        self.model_save_path = os.path.join("models_save",self.base_save_path, self.model_save_path_base)
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)


        self.args.observation_dim = env.observation_dim
        self.args.avail_actions_dim = env.avail_actions_dim
        self.args.state_dim = env.state_dim
        self.args.agent_net_hidden_dim = args.agent_net_hidden_dim
        self.args.qmix_hidden_dim = args.qmix_hidden_dim
        self.args.hyper_hidden_dim = args.hyper_hidden_dim

        self.qmix_algo = QMixAlgo(self.args)
        self.replay_buffer = DelayBuffer(self.args)

        self.show_plot = bool(args.show_plot)

        # 记录训练信息
        self.evaluate_info = {
            "loss_list": [],
            "reward_list": [],
            "epsilon_list": [],
            "server_content_list": [],
            "step_list": [],
        }

    def run(self):
        """
        训练QMIX算法
        :return:
        """

        # 需要记录的指标
        reward_collection = []    # 收集一个self.evaluate_freq的奖励
        loss_collection = []      # 收集一个self.evaluate_freq的损失


        while self.total_episodes < self.max_episodes:

            # 记录阶段训练信息
            if (self.total_episodes+1) % self.evaluate_freq == 0:
                self.evaluate_info["loss_list"].append(np.mean(loss_collection))
                self.evaluate_info["reward_list"].append(np.mean(reward_collection))
                reward_collection = []
                loss_collection = []

            self.env.reset()

            # 记录全局训练信息
            # 下降epsilon
            self.evaluate_info["epsilon_list"].append(self.epsilon)
            self.epsilon = self.decay_epsilon()

            step = 0
            # 训练
            for step in range(self.max_steps):
                last_state = self.env.get_state()
                last_observation = self.env.get_observations()
                avail_actions = self.env.get_avail_actions()

                # step
                actions = self.qmix_algo.choose_action(last_observation, avail_actions, self.epsilon)
                rewards, dones = self.env.step(actions)

                reward_collection.append(rewards)

                next_state = self.env.get_state()
                next_observation = self.env.get_observations()
                next_avail_actions = self.env.get_avail_actions()
                terminate = 1 if self.env.is_done() else 0

                # 存储数据
                self.replay_buffer.store_transition(
                    last_states=last_state, last_observations=last_observation,
                    actions=actions, rewards=rewards, dones=dones,
                    next_states=next_state, next_observations=next_observation,
                    avail_actions=avail_actions, next_avail_actions=next_avail_actions,
                    terminate=terminate,
                )

                # 训练网络
                if self.replay_buffer.memory_size > self.batch_size:
                    batch_data = self.replay_buffer.sample(self.batch_size)
                    loss = self.qmix_algo.train(batch_data, self.total_episodes).detach().cpu().numpy()

                    loss_collection.append(loss)



                if self.env.is_done() or step == self.max_steps-1:
                    self.evaluate_info["step_list"].append(step)
                    self.evaluate_info["server_content_list"].append(self.env.get_server_content())
                    break

            print(f"Episode:{self.total_episodes+1}\tStep:{step}, " + \
                  f"Reward.mean: {np.mean(reward_collection):.6f}, Loss.mean: {np.mean(loss_collection):.6f}")
            self.total_episodes += 1
        self.save_model_info()
        self.test()
        pass

    def save_model_info(self):
        """
        保存模型信息
        指标：
            1. 奖励函数 曲线
            2. 损失函数 曲线
            3. epsilon下降曲线
            4. 模型保存
        :return:
        """
        # 保存训练信息
        for key, value in self.evaluate_info.items():
            fig = plt.figure()
            plt.plot(value)
            plt.title(key)
            plt.savefig(os.path.join(self.model_save_path, key+'.png'))
            if self.show_plot:
                plt.show()
            plt.close(fig)

        # 画出步数与收集数据量的关系
        fig = plt.figure()
        plt.plot(self.evaluate_info["step_list"], 'r-')
        plt.plot(self.evaluate_info["server_content_list"], 'b-')
        plt.legend(["Step", "Server Content"])
        plt.title("Step and Server Content")
        plt.xlabel("Episode")
        plt.savefig(os.path.join(self.model_save_path, "step_server_content.png"))
        if self.show_plot:
            plt.show()
        plt.close(fig)


        # 保存模型
        self.qmix_algo.save_model(self.model_save_path)

        # 将参数写入文件
        with open(self.model_save_path + "/parameters.txt", "w") as f:
            for arg in vars(self.args):
                f.write(f"{arg}: {getattr(self.args, arg)}\n")
        pass

    def test(self):
        """
        测试模型: 执行10次网络确定的policy self.epsilon = 0.0
        画出找到旗子的数量折线图
        :return:
        """
        print("this is a test function.")
        self.env.reset()
        data_content_list = []
        step = 0
        while True:
            last_observations = self.env.get_observations()
            last_states = self.env.get_state()
            avail_actions = self.env.get_avail_actions()
            actions = self.qmix_algo.choose_action(last_observations, avail_actions, epsilon=0.0)
            dones, rewards = self.env.step(actions)
            next_avail_actions = self.env.get_avail_actions()
            next_observations = self.env.get_observations()
            next_states = self.env.get_state()
            terminate = 1 if self.env.is_done() else 0
            self.replay_buffer.store_transition(last_states=last_states, last_observations=last_observations,
                    actions=actions, rewards=rewards, dones=dones,
                    next_states=next_states, next_observations=next_observations,
                    avail_actions=avail_actions, next_avail_actions=next_avail_actions,
                    terminate=terminate,)
            # 数据服务器容量数量
            data_content_list.append(self.env.get_server_content())
            step += 1
            if step == self.max_steps or self.env.is_done():
                break

        plt.plot(list(range(step)), data_content_list)
        plt.xlabel('step')
        plt.ylabel('data_content')
        plt.title("Content Result")
        plt.savefig(self.model_save_path + "/test_result.png")
        if self.show_plot:
            plt.show()
        plt.close()
        print(f"test end!")

    def decay_epsilon(self):
        """
        epsilon下降函数
        :return:
        """
        decay_func = {
            "function1": "np.log(2) * self.total_episodes / self.epsilon_decay_steps",
            "function2": "np.power(np.log(2), 1/2) * self.total_episodes / self.epsilon_decay_steps",
            "function3": "np.power(np.log(2), 1/5) * self.total_episodes / self.epsilon_decay_steps",
        }
        if self.epsilon_decay_function == "function1":
            exp_x = np.log(2) * self.total_episodes / self.max_episodes
            epsilon = max(-np.exp(exp_x) + 2, self.epsilon_min)
        elif self.epsilon_decay_function == "function2":
            exp_x = np.power(np.log(2), 1/2) * self.total_episodes / self.max_episodes
            epsilon = max(-np.exp(exp_x) + 2, self.epsilon_min)
        elif self.epsilon_decay_function == "function3":
            exp_x = np.power(np.log(4), 1) * self.total_episodes / self.max_episodes
            epsilon = max(-np.exp(exp_x) + 4, self.epsilon_min)
        elif self.epsilon_decay_function == "default":
            epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        epsilon = min(1, epsilon)
        return epsilon

    def test_decay_epsilon(self,):
        """
        画出epsilon下降曲线
        :return:
        """
        epsilon_list = []
        for self.total_episodes in range(self.max_episodes):
            self.epsilon = self.decay_epsilon()
            epsilon_list.append(self.epsilon)
        fig = plt.figure()
        plt.plot(epsilon_list)
        plt.title("Epsilon Decay Curve")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.savefig(os.path.join(self.model_save_path, "epsilon_decay.png"))
        if self.show_plot:
            plt.show()
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters setting for QMIX.")
    parser.add_argument("--max_episodes", type=int, default=int(1000), help=" Maximum number of training steps")
    parser.add_argument("--max_steps", type=int, default=int(50), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=int, default=2, help="Evaluate the policy every 'evaluate_freq' steps")

    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_decay_steps", type=float, default=800, help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_function", type=str, default="default",
                        help="epsilon_decay_function", choices=["default", "function1", "function2", "function3"])

    parser.add_argument("--buffer_size", type=int, default=50, help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="use lr decay")
    parser.add_argument("--gamma", type=float, default=0.77, help="Discount factor")

    parser.add_argument("--qmix_hidden_dim", type=int, default=640, help="The dimension of the hidden layer of the QMIX network")
    parser.add_argument("--hyper_hidden_dim", type=int, default=640, help="The dimension of the hidden layer of the hyper-network")
    parser.add_argument("--agent_net_hidden_dim", type=int, default=640, help="The dimension of the hidden layer of the agent network")

    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network")
    parser.add_argument("--seed", type=int, default=44, help="Random seed")
    parser.add_argument("--use_RMS", type=bool, default=False, help="Use RMS")
    parser.add_argument("--show_plot", type=bool, default=False, help="Show plot", choices=[True, False])
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Use orthogonal initialization")

    # 环境参数
    parser.add_argument("--env_title", type=str, default="Environment", help="The title of the environment")
    parser.add_argument("--render_mode", type=str, default="None", choices=["human", "None"], help="Render mode")
    parser.add_argument("--n_phone", type=int, default=1, help="The number of phones")
    parser.add_argument("--n_uav", type=int, default=1, help="The number of uavs")
    parser.add_argument("--n_flag", type=int, default=5, help="The number of flags")
    parser.add_argument("--n_obstacle", type=int, default=2, help="The number of obstacles")
    parser.add_argument("--n_server", type=int, default=1, help="The number of servers")
    parser.add_argument("--agent_vision_length", type=int, default=3, help="The length of agent vision")
    parser.add_argument("--padding", type=int, default=4, help="The padding of the agent")
    parser.add_argument("--width", type=int, default=5, help="The width of the agent")
    parser.add_argument("--height", type=int, default=5, help="The height of the agent")
    parser.add_argument("--use_init_position", type=bool, default=False, help="Use initial position")
    parser.add_argument("--use_object_one_hot", type=bool, default=True, help="Use object one hot")
    parser.add_argument("--use_uav_content", type=bool, default=True, help="Use uav content")

    parser.add_argument("--base_save_path", type=str, default=".", help="The path of the comparison saved model")



    args = parser.parse_args()

    args.epsilon_decay = (args.epsilon - args.epsilon_min) / args.epsilon_decay_steps

    begin_time = time.time()

    # Initialize environment
    env = Environment(args)

    # 运行算法
    runer_qmix = Runner_QMix(env, args)
    runer_qmix.run()

    end_time = time.time()
    print("Training finished!\n")
    print("Total time: ", end_time - begin_time)

    # 保存信息
    with open(os.path.join(runer_qmix.model_save_path, runer_qmix.train_info_file), "a") as f:

        f.write(f"训练时间:{end_time-begin_time:.2f}s\n")
        f.close()


