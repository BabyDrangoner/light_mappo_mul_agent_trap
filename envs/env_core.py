import numpy as np


class EnvCoreBase(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
        self.obs_dim = 14  # 设置智能体的观测维度 # set the observation dimension of agents
        self.action_dim = 5  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14,))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,)))
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]


class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self):
        # system parameter
        self.targets_num = 1
        self.row = 20
        self.col = 20
        self.t = 0
        self.t_total = 100
        self.safe_dis = 1.5
        self.agent_num = 1 # 设置智能体(小飞机)的个数，这里设置为两个 # set the number of agents(aircrafts), here set to two
        self.obs_dim = 4  # 设置智能体的观测维度 # set the observation dimension of agents
        self.action_dim = 2  # 设置智能体的动作维度，这里假定为一个五个维度的 # set the action dimension of agents, here set to a five-dimensional
        self.delta_t = 0.2
        # Location
        self.agents_location = np.zeros((2, self.agent_num))
        self.targets_location = np.zeros((2, self.targets_num))

        # Velocity
        self.agents_velocity = np.zeros((2, self.agent_num))
        self.agents_accelerate = np.zeros((2, self.agent_num))
        self.targets_velocity = np.zeros((2, self.targets_num))

        # Distance
        self.distance_each_agent = np.zeros((self.agent_num, self.agent_num))
        self.distance_each_target = np.zeros((self.agent_num, self.targets_num))

        # Theta
        self.agents_theta = np.zeros((self.agent_num, self.agent_num))
        self.theta_standard = np.pi * 2 / self.agent_num

        # Agents_theta
        self.agents_theta_list = []

        # Agents_distance
        self.agents_distance_list = {}

    def reset(self):
        """
        其他部分：

        """
        self.t = 0
        self.targets_location[0, 0] = 10
        self.targets_location[1, 0] = 10
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        # When self.agent_num is set to 2 agents, the return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        sub_agent_obs = []
        # for i in range(self.agent_num):
        #     sub_obs = np.random.random(size=(14,))
        #     sub_agent_obs.append(sub_obs)

        for i in range(self.agent_num):
            theta = np.random.uniform(0, 2 * np.pi)
            self.agents_location[0, i] = 10 + 10 * np.cos(theta)
            self.agents_location[1, i] = 10 + 10 * np.sin(theta)
            self.agents_velocity[0, i] = 0.0
            self.agents_velocity[1, i] = 0.0
            self.agents_accelerate[0, i] = 0.0
            self.agents_accelerate[1, i] = 0.0
            # 计算位置差和速度差，确保它们保持二维形状
            position_diff = (self.targets_location[:, 0] - self.agents_location[:, i]).reshape(-1)
            velocity_diff = (self.targets_velocity[:, 0] + self.agents_velocity[:, i]).reshape(-1)
            # 合并差值作为观察结果
            sub_obs = np.concatenate((position_diff, velocity_diff), axis=-1)
            # sub_obs = np.concatenate((sub_obs, self.agents_accelerate[:, i].reshape(-1)), axis=-1)
            self.distance_each_target[i, 0] = np.linalg.norm(position_diff)
            sub_agent_obs.append(sub_obs)

        return sub_agent_obs

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list，每个list里面为一个shape = (self.action_dim, )的动作数据
        # 默认参数情况下，输入为一个list，里面含有两个元素，因为动作维度为5，所里每个元素shape = (5, )
        # When self.agent_num is set to 2 agents, the input of actions is a 2-dimensional list, each list contains a shape = (self.action_dim, ) action data
        # The default parameter situation is to input a list with two elements, because the action dimension is 5, so each element shape = (5, )
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        # for i in range(self.agent_num):
        #     sub_agent_obs.append(np.random.random(size=(14,)))
        #     sub_agent_reward.append([np.random.rand()])
        #     sub_agent_done.append(False)
        #     sub_agent_info.append({})

        # Calculate the next obs
        for i in range(self.agent_num):
            action = actions[i]
            self.agents_accelerate[0, i] = action[0]
            self.agents_accelerate[1, i] = action[1]
            self.agents_velocity[:, i] += self.agents_accelerate[:, i] * self.delta_t  # todo
            self.agents_velocity[:, i] = np.clip(self.agents_velocity[:, i], -3, 3)

            self.agents_location[:, i] += self.agents_velocity[:, i] * self.delta_t
            self.agents_location[:, i] = np.clip(self.agents_location[:, i], 0, self.col)

            position_diff = (self.targets_location[:, 0] - self.agents_location[:, i]).reshape(-1)
            velocity_diff = (self.targets_velocity[:, 0] + self.agents_velocity[:, i]).reshape(-1)
            sub_obs = np.concatenate((position_diff, velocity_diff), axis=-1)
            # sub_obs = np.concatenate((sub_obs, self.agents_accelerate[:, i].reshape(-1)), axis=-1)
            sub_agent_obs.append(sub_obs)
        # sub_agent_info.append(np.concatenate((self.agents_location, self.targets_location), axis=1))


        # Calculate the reward
        for k in range(self.agent_num):
            reward = 0
            # theta = 0
            # for i in range(self.agent_num):
            #     if i != k:
            #         self.distance_each_agent[k, i] = np.linalg.norm(self.agents_location[:, k]
            #                                                         - self.agents_location[:, i])
            #         delta_dis = self.distance_each_agent[k, i] - self.safe_dis
            #         reward += min(delta_dis, 0) * 2
            #         # 与目标的相对位置   todo
            #         vector1 = self.agents_location[:, k] - self.targets_location[:, 0]
            #         vector2 = self.agents_location[:, i] - self.targets_location[:, 0]
            #         # 相对角度
            #         theta0 = self.calculate_angle_between_agents_radians(vector1, vector2)
            #         reward -= abs(self.theta_standard - theta0)
            #         theta += theta0
            #         self.agents_theta[k][i] = theta0

            # self.agents_theta_list[k].append(theta / 2)

            for i in range(self.targets_num):
                self.distance_each_target[k, i] = np.linalg.norm(self.agents_location[:, k]
                                                                 - self.targets_location[:, i])
                # self.agents_distance_list[k].append(self.distance_each_target[k, i])
                # reward += distance0 - self.distance_each_target[k, i]
                reward -= self.distance_each_target[k, i]

                # delta_dis = self.distance_each_target[k, i] - self.safe_dis * 1.5
                # reward += min(delta_dis, 0) * 2

            sub_agent_reward.append([reward])

        # Calculate the done
        self.t += 1
        if self.t == self.t_total:
            for i in range(self.agent_num):
                sub_agent_done.append(True)
            return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

        done = True
        for k in range(self.agent_num):
            if done:
                for i in range(self.targets_num):
                    if done and self.distance_each_target[k, i] > 0.1:
                        done = False
                        break


            sub_agent_done.append(done)

        if done:
            for k in range(self.agent_num):
                sub_agent_reward[k][0] += 100
            print("Now task is over !")

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]

    # 其他函数
    def calculate_angle_between_agents_radians(self, vector_a, vector_b):
        """
        计算两个向量之间的夹角（以弧度为单位）。

        参数:
        - vector_a: 智能体A相对于目标的位置矢量。
        - vector_b: 智能体B相对于目标的位置矢量。

        返回:
        - angle_rad: 两个向量之间的夹角（弧度）。
        """
        # 计算点积
        dot_product = np.dot(vector_a, vector_b)

        # 计算两个向量的模
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        # 计算余弦值
        cos_angle = dot_product / (norm_a * norm_b)

        # 为了防止因浮点运算误差导致的cos_angle微小超出[-1, 1]的范围，
        # 使用clip函数限制cos_angle的值在这个范围内
        cos_angle = np.clip(cos_angle, -1, 1)

        # 计算夹角（以弧度为单位）
        angle_rad = np.arccos(cos_angle)

        return angle_rad