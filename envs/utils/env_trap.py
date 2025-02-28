import numpy as np
from .env_base import EnvBase
from .agent import RobotManager
from config import get_config


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default="my_env", help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=3, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def calculate_angle_between_agents_radians(vector_a, vector_b):
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


class TaskHandler:
    def __init__(self, trap_robots_nums, target_robots_nums,
                 safe_dis, trap_dis,
                 approach_reward_w, collision_reward_w, theta_reward_w, done_reward):
        self.m_trap_robots_nums = trap_robots_nums
        self.m_target_robots_nums = target_robots_nums
        self.m_old_info = []
        self.m_robot_target = []
        self.m_type = None

        self.m_safe_dis = safe_dis
        self.m_trap_dis = trap_dis
        self.m_final_theta = np.pi * 2 / trap_robots_nums
        self.m_init_center = [10, 10]

        self.m_approach_reward_w = approach_reward_w
        self.m_collision_reward_w = collision_reward_w
        self.m_theta_reward_w = theta_reward_w
        self.m_done_reward = done_reward

        self.m_state = None
        self.m_reward = None
        self.m_done = None

    def get_init_state(self):
        init_location = {"target": [np.array([self.m_init_center]).reshape(-1, 1)], "trap": []}
        init_yaw_angle = {"target": [np.pi / 2], "trap": [np.pi / 2 for _ in range(self.m_trap_robots_nums)]}

        for _ in range(self.m_trap_robots_nums):
            theta = np.random.rand() * np.pi * 2
            init_location["trap"].append(np.array([self.m_init_center[0] * np.cos(theta),
                                                   self.m_init_center[1] * np.sin(theta)]).reshape(-1, 1))
        return init_location, init_yaw_angle

    def reset(self, target_choice, type, robots_manager):

        self.m_type = type
        self.m_old_info = [{} for _ in range(self.m_trap_robots_nums)]
        self.m_robot_target = target_choice.copy()

        self.set_old_info(robots_manager, type)

        return self.m_state.values()

    def get_reward_done(self, robots_manager, type):
        self.set_old_info(robots_manager, type)
        return [self.m_state.values(), self.m_reward.values(), self.m_done.values(), None]

    def set_old_info(self, robots_manager, type):
        if self.m_type != type:
            self.m_old_info = [{} for _ in range(robots_manager.get_type_robots_nums[type])]
            self.m_type = type

        self.m_state = {}
        self.m_reward = {}
        self.m_done = {}
        done = True

        robots = robots_manager.get_type_all_robots(type)
        for i, robot in enumerate(robots):
            old_info = self.m_old_info[i]

            r_la = robot.get_linear_accelerate()
            r_aa = robot.get_angular_accelerate()
            r_lv = robot.get_linear_velocity()
            r_av = robot.get_angular_velocity()
            r_lc = robot.get_location()

            target_robot = robots_manager.get_robot("target", self.m_robot_target[i])  # todo
            t_lv = target_robot.get_linear_velocity()
            t_av = target_robot.get_angular_velocity()
            t_lc = target_robot.get_location()

            tr_lc = (t_lc - r_lc).reshape(-1)
            tr_lv = (t_lv - r_lv)
            tr_av = (t_av - r_av)
            try:
                old_dis = old_info["distance_between_robot_target"]
            except:
                old_dis = 0
            new_dis = np.linalg.norm(tr_lc)
            old_info["distance_between_robot_target"] = new_dis

            r_name = str(robot.get_id())
            self.m_state[r_name] = np.concatenate((tr_lc, np.array([tr_lv, tr_av]).reshape(-1)), axis=0)
            self.m_state[r_name] = np.concatenate((self.m_state[r_name], np.array([r_la, r_aa]).reshape(-1)), axis=0)

            self.m_reward[r_name] = 0
            self.m_reward[r_name] += old_dis - new_dis

            self.m_done[r_name] = False

            if abs(new_dis - self.m_trap_dis) > 0.05:
                done = False

            for j, robot_j in enumerate(robots):
                if j == i:
                    continue

                rj_lv = robot_j.get_linear_velocity()
                rj_av = robot_j.get_angular_velocity()
                rj_lc = robot_j.get_location()

                r_ij_lc = (rj_lc - r_lc).reshape(-1)
                r_ij_lv = (rj_lv - r_lv)
                r_ij_av = (rj_av - r_av)

                tmp = np.concatenate((r_ij_lc, np.array([r_ij_lv, r_ij_av]).reshape(-1)), axis=0)
                self.m_state[r_name] = np.concatenate((self.m_state[r_name], tmp), axis=0)

                r_ij_dis = np.linalg.norm(r_ij_lc)
                self.m_reward[r_name] += self.m_collision_reward_w * min(0, r_ij_dis - self.m_safe_dis)

                target_robot = robots_manager.get_robot("target", self.m_robot_target[j])  # todo
                t_lc = target_robot.get_location()
                trj_lc = (t_lc - rj_lc).reshape(-1)

                theta = calculate_angle_between_agents_radians(trj_lc, tr_lc)
                delta_theta = abs(theta - self.m_final_theta)
                self.m_reward[r_name] -= self.m_theta_reward_w * delta_theta

                if delta_theta > 0.01:
                    done = False

        if ~done:
            return

        for robot in robots:
            r_name = str(robot.get_id())
            self.m_done[r_name] = done
            self.m_reward[r_name] += self.m_done_reward
        print("task is over")


class EnvTrap(EnvBase):
    def __init__(self):
        # 获取配置参数
        args = get_config()
        self.all_args = args.parse_args()

        super().__init__(self.all_args.max_col, self.all_args.max_row, self.all_args.delta_t, self.all_args.t_total)

        self.agent_num = self.all_args.trap_robots_nums
        self.obs_dim = (2 + 1 + 1) * self.all_args.trap_robots_nums + 1 + 1
        self.action_dim = 2

        self.m_robot_manager = RobotManager()
        self.init_robots()

        self.m_TaskHandler = TaskHandler(
            self.all_args.trap_robots_nums, self.all_args.target_robots_nums,
            self.all_args.safe_dis, self.all_args.trap_dis,
            self.all_args.approach_reward_w, self.all_args.collision_reward_w,
            self.all_args.theta_reward_w, self.all_args.done_reward
        )

    def get_robot_manager(self):
        return self.m_robot_manager

    def init_robots(self):
        '''
        id, size, type,
        max_linear_accelerate, max_angular_accelerate,
        max_linear_velocity, max_angular_velocity
        '''
        types = ["target", "trap"]
        robots_params = {"target": [(0, 1, "target", 0, 0, 0, 0)]}

        robots_params["trap"] = []
        for i in range(self.agent_num):
            robots_params["trap"].append((i, 1, "trap"
                                          , self.all_args.max_linear_accelerate, self.all_args.max_angular_accelerate
                                          , self.all_args.max_linear_velocity, self.all_args.max_angular_velocity))
        self.m_robot_manager.init_robots(types, robots_params)

    def reset(self):
        self.base_reset()

        robots_init_location, robots_init_yawAngle = self.m_TaskHandler.get_init_state()
        self.m_robot_manager.robots_reset(robots_init_location, robots_init_yawAngle)

        target_id = self.m_robot_manager.get_type_all_robots("target")[0].get_id()
        target_choice = [target_id for _ in range(len(robots_init_location["trap"]))]
        return self.m_TaskHandler.reset(target_choice, "trap", self.m_robot_manager)

    def step(self, actions):

        self.base_step()
        linear_accelerate = {"target": [0], "trap": [action[0] for action in actions]}
        angular_accelerate = {"target": [0], "trap": [action[1] for action in actions]}

        self.m_robot_manager.robots_action(linear_accelerate, angular_accelerate, self.m_delta_t)

        for robot in self.m_robot_manager.get_all_robots():
            location = robot.get_location()
            robot.set_location(location)

        return self.m_TaskHandler.get_reward_done(self.m_robot_manager, "trap")

    def modify_location(self, location):
        location[0] = np.clip(location[0], 0, self.m_max_col)
        location[1] = np.clip(location[1], 0, self.m_max_row)





