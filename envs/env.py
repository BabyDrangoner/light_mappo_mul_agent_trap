import numpy as np
from utils.env_trap import calculate_angle_between_agents_radians
class EnvV2:
    def __init__(self, agents_num, targets_num, row, col, safe_dis):
        # system parameter
        self.agents_num = agents_num
        self.targets_num = targets_num
        self.row = row
        self.col = col
        self.t = 0
        self.t_total = 100
        self.safe_dis = safe_dis
        self.agent_types = [str(i) for i in range(agents_num)]
        self.action_continue = True
        self.action_dim = 2
        self.obsp_dim = 14

        # Location
        self.agents_location = np.zeros((2, agents_num))
        self.targets_location = np.zeros((2, targets_num))

        # Velocity
        self.agents_velocity = np.zeros((2, agents_num))
        self.agents_accelerate = np.zeros((2, agents_num))
        self.targets_velocity = np.zeros((2, targets_num))

        # Distance
        self.distance_each_agent = np.zeros((agents_num, agents_num))
        self.distance_each_target = np.zeros((agents_num, targets_num))

        # Theta
        self.agents_theta = np.zeros((agents_num, agents_num))
        self.theta_standard = np.pi * 2 / self.agents_num

        # Choice
        self.agents_choice = np.zeros(agents_num)

        # Agents_name
        self.agents_name = [str(i) for i in range(agents_num)]

        # Agents_theta
        self.agents_theta_list = {}
        for i in range(self.agents_num):
            self.agents_theta_list[self.agents_name[i]] = []

        # Agents_distance
        self.agents_distance_list = {}
        for i in range(self.agents_num):
            self.agents_distance_list[self.agents_name[i]] = []

    def reset(self):
        agents_obs = {}
        self.t = 0
        self.targets_location[0, 0] = 10
        self.targets_location[1, 0] = 10
        for i in range(self.agents_num):
            theta = np.random.uniform(0, 2 * np.pi)
            self.agents_location[0, i] = self.targets_location[0, 0] + 10 * np.cos(theta)
            self.agents_location[1, i] = self.targets_location[1, 0] + 10 * np.sin(theta)
            self.agents_velocity[0, i] = 0.0
            self.agents_velocity[1, i] = 0.0
            self.agents_accelerate[0, i] = 0.0
            self.agents_accelerate[1, i] = 0.0
            # 计算位置差和速度差，确保它们保持二维形状
            position_diff = (self.targets_location[:, 0] - self.agents_location[:, i]).reshape(-1, 1)
            velocity_diff = (self.targets_velocity[:, 0] - self.agents_velocity[:, i]).reshape(-1, 1)
            # 合并差值作为观察结果
            agents_obs[self.agents_name[i]] = np.concatenate((position_diff, velocity_diff), axis=1)
            agents_obs[self.agents_name[i]] = np.concatenate((agents_obs[self.agents_name[i]],
                                                              self.agents_accelerate[:, i].reshape(-1, 1)), axis=1)
            self.distance_each_target[i, 0] = np.linalg.norm(position_diff)

        for i in range(self.agents_num):
            for j in range(self.agents_num):
                if i != j:
                    position_diff = (self.agents_location[:, i] - self.agents_location[:, j]).reshape(-1, 1)
                    velocity_diff = (self.agents_velocity[:, i] - self.agents_velocity[:, j]).reshape(-1, 1)
                    agents_obs[self.agents_name[i]] = np.concatenate((agents_obs[self.agents_name[i]], position_diff),
                                                                     axis=1)
                    agents_obs[self.agents_name[i]] = np.concatenate((agents_obs[self.agents_name[i]], velocity_diff),
                                                                     axis=1)
                    self.distance_each_agent[i, j] = np.linalg.norm(position_diff)

            agents_obs[self.agents_name[i]] = np.array([agents_obs[self.agents_name[i]]])

        return agents_obs


    def step(self, agents_action, delta_t):  # next_obs, reward, done
        agents_next_obs = {}
        agents_reward = {}
        agents_done = {}  # todo

        # Calculate the next obs
        for i in range(self.agents_num):
            action = agents_action[0][i]
            self.agents_velocity[0, i] += action[0] * delta_t  # todo
            self.agents_velocity[1, i] += action[1] * delta_t
            self.agents_location[:, i] += self.agents_velocity[:, i] * delta_t
            position_diff = (self.targets_location[:, 0] - self.agents_location[:, i]).reshape(-1, 1)
            velocity_diff = (self.targets_velocity[:, 0] - self.agents_velocity[:, i]).reshape(-1, 1)
            agents_next_obs[self.agents_name[i]] = np.concatenate((position_diff, velocity_diff), axis=1)
            agents_next_obs[self.agents_name[i]] = np.concatenate((agents_next_obs[self.agents_name[i]],
                                                                   self.agents_accelerate[:, i].reshape(-1, 1)), axis=1)
        for i in range(self.agents_num):
            for j in range(self.agents_num):
                if i != j:
                    position_diff = (self.agents_location[:, i] - self.agents_location[:, j]).reshape(-1, 1)
                    velocity_diff = (self.agents_velocity[:, i] - self.agents_velocity[:, j]).reshape(-1, 1)
                    agents_next_obs[self.agents_name[i]] = np.concatenate((agents_next_obs[self.agents_name[i]],
                                                                           position_diff), axis=1)
                    agents_next_obs[self.agents_name[i]] = np.concatenate((agents_next_obs[self.agents_name[i]],
                                                                           velocity_diff), axis=1)
            agents_next_obs[self.agents_name[i]] = np.array([agents_next_obs[self.agents_name[i]]])

        # Calculate the reward
        for k in range(self.agents_num):
            reward = 0
            theta = 0
            for i in range(self.agents_num):
                if i != k:
                    self.distance_each_agent[k, i] = np.linalg.norm(self.agents_location[:, k]
                                                                    - self.agents_location[:, i])
                    delta_dis = self.distance_each_agent[k, i] - self.safe_dis
                    reward += min(delta_dis, 0) * 2
                    # 与目标的相对位置   todo
                    vector1 = self.agents_location[:, k] - self.targets_location[:, 0]
                    vector2 = self.agents_location[:, i] - self.targets_location[:, 0]
                    # 相对角度
                    theta0 = calculate_angle_between_agents_radians(vector1, vector2)
                    reward -= abs(self.theta_standard - theta0)
                    theta += theta0
                    self.agents_theta[k][i] = theta0

            self.agents_theta_list[self.agents_name[k]].append(theta / 2)

            for i in range(self.targets_num):
                distance0 = self.distance_each_target[k, i]
                self.distance_each_target[k, i] = np.linalg.norm(self.agents_location[:, k]
                                                                 - self.targets_location[:, i])
                self.agents_distance_list[self.agents_name[k]].append(self.distance_each_target[k, i])
                reward += distance0 - self.distance_each_target[k, i]
                delta_dis = self.distance_each_target[k, i] - self.safe_dis * 1.5
                reward += min(delta_dis, 0) * 2
                if self.distance_each_target[k, i] > 12:
                    reward -= 10
            agents_reward[self.agents_name[k]] = reward

        # Calculate the done
        self.t += 1
        if self.t == self.t_total:
            for i in range(self.agents_num):
                agents_done[self.agents_name[i]] = True
            return agents_next_obs, agents_reward, agents_done
        done = True
        for k in range(self.agents_num):
            if done:
                for i in range(self.targets_num):
                    if done and abs(self.distance_each_target[k, i] - self.safe_dis * 1.5) > 0.01:
                        done = False
                        break
                if done:
                    for i in range(self.agents_num):
                        if k != i:
                            if self.distance_each_agent[k, i] < self.safe_dis:
                                done = False
                                break
                            if abs(self.agents_theta[k][i] - self.theta_standard) < 0.01:
                                done = False
                                break

            agents_done[self.agents_name[k]] = done

        if done:
            for k in range(self.agents_num):
                agents_reward[self.agents_name[k]] += 10

        return agents_next_obs, agents_reward, agents_done