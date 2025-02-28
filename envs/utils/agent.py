import numpy as np


class RobotBase:
    def __init__(self, id, size,
                 max_linear_accelerate, max_angular_accelerate, max_linear_velocity, max_angular_velocity,
                 ):

        # id
        self.m_id = id

        # size
        self.m_size = size

        # location
        self.m_location = np.zeros((2, 1))

        # yawn angle
        self.m_yawAngle = 0

        # accelerate
        self.m_linearAccelerate = 0
        self.m_max_linear_accelerate = max_linear_accelerate
        self.m_angular_accelerate = 0
        self.m_max_angular_accelerate = max_angular_accelerate

        # velocity
        self.m_linear_velocity = 0
        self.m_max_linear_velocity = max_linear_velocity
        self.m_angular_velocity = 0
        self.m_max_angular_velocity = max_angular_velocity

    def reset(self, init_location, init_yawAngle):
        # location
        self.set_location(init_location)

        # yawn angle
        self.set_yaw_angle(init_yawAngle)

        # accelerate
        self.set_linear_accelerate(0)
        self.set_angular_accelerate(0)

        # velocity
        self.set_linear_velocity(0)
        self.set_angular_velocity(0)

    def action(self, linear_accelerate, angular_accelerate, delta_t):
        # 设置加速度
        self.set_linear_accelerate(linear_accelerate)
        self.set_angular_accelerate(angular_accelerate)

        # 在机器人基类中，速度不加限制，由子类或者控制器控制速度界限
        self.m_linear_velocity += self.m_linear_accelerate * delta_t
        self.m_angular_velocity += self.m_angular_accelerate * delta_t

        # 运动
        self.m_yawAngle += self.m_angular_velocity * delta_t
        while self.m_yawAngle < 0:
            self.m_yawAngle += np.pi * 2
        self.m_yawAngle %= np.pi * 2

        self.m_location[0] += self.m_linear_velocity * np.cos(self.m_yawAngle)
        self.m_location[1] += self.m_linear_velocity * np.sin(self.m_yawAngle)

    def get_id(self):
        return self.m_id

    def set_id(self, v):
        self.m_id = v

    def get_size(self):
        return self.m_size

    def set_size(self, v):
        self.m_size = v

    def get_linear_accelerate(self):
        return self.m_linear_accelerate

    def set_linear_accelerate(self, v):
        self.m_linear_accelerate = min(self.m_max_linear_accelerate, max(0, v))

    def get_angular_accelerate(self):
        return self.m_angular_accelerate

    def set_angular_accelerate(self, v):
        self.m_angular_accelerate = min(self.m_max_angular_accelerate, max(0, v))

    def get_linear_velocity(self):
        return self.m_linear_velocity

    def set_linear_velocity(self, v):
        self.m_linear_velocity = v

    def get_angular_velocity(self):
        return self.m_angular_velocity

    def set_angular_velocity(self, v):
        self.m_angular_velocity = v

    def get_yaw_angle(self):
        return self.m_yawAngle

    def set_yaw_angle(self, v):
        self.m_yawAngle = v

    def get_location(self):
        return self.m_location

    def set_location(self, v):
        self.m_location[0][0] = v[0]   # 防止浅拷贝(或许不存在这种情况)
        self.m_location[1][0] = v[1]


class Robot(RobotBase):
    def __init__(self, id, size, type,
                 max_linear_accelerate, max_angular_accelerate,
                 max_linear_velocity, max_angular_velocity,):
        super().__init__(id, size,
                         max_linear_accelerate, max_angular_accelerate,
                         max_linear_velocity, max_angular_velocity,)
        self.type = type

    def get_type(self):
        return self.type

    def set_type(self, v):
        self.type = v


class RobotManager:
    def __init__(self):
        self.m_types = []
        self.m_robot_map = {}
        self.m_robot_nums = {}

    def clear_all(self):
        self.m_types.clear()
        self.m_robot_map.clear()
        self.m_robot_nums.clear()

    def init_robots(self, types, robots_params):
        self.clear_all()

        for type in types:
            self.m_types.append(type)
            self.m_robot_map[type] = []

            info = robots_params[type]
            self.m_robot_nums[type] = len(info)
            for param in robots_params[type]:
                robot = Robot(*param)
                self.m_robot_map[type].append(robot)

    def robots_reset(self, init_locations, init_yawAngles):
        for type in init_locations.keys():
            locations = init_locations[type]
            yawAngles = init_yawAngles[type]
            for index, robot in enumerate(self.m_robot_map[type]):
                robot.reset(locations[index], yawAngles[index])

    def robots_action(self, linear_accelerates, angular_accelerates, delta_t):
        for type in linear_accelerates.keys():
            las = linear_accelerates[type]
            aas = angular_accelerates[type]
            for index, robot in enumerate(self.m_robot_map[type]):
                robot.action(las[index], aas[index], delta_t)

    def get_robot(self, type, id):
        return self.m_robot_map[type][id]

    def get_type_all_robots(self, type):
        return self.m_robot_map[type]

    def get_all_robots(self):
        robots = []
        for keys, values in self.m_robot_map.items():
            for robot in values:
                robots.append(robot)
        return robots

    def get_type_robots_nums(self, type):
        return self.m_robot_nums[type]

    def get_all_robots_num(self):
        return self.m_robot_nums

    def robot_set_id(self, type, id, v):
        self.m_robot_map[type][id].set_id(v)

    def robot_get_size(self, type, id):
        return self.m_robot_map[type][id].get_size()

    def robot_set_size(self, type, id, v):
        self.m_robot_map[type][id].set_size(v)

    def robot_get_linear_accelerate(self, type, id):
        return self.m_robot_map[type][id].get_linear_accelerate()

    def robot_set_linear_accelerate(self, type, id, v):
        self.m_robot_map[type][id].set_linear_accelerate(v)

    def robot_get_angular_accelerate(self, type, id):
        return self.m_robot_map[type][id].get_angular_accelerate()

    def robot_set_angular_accelerate(self, type, id, v):
        self.m_robot_map[type][id].set_angular_accelerate(v)

    def robot_get_linear_velocity(self, type, id):
        return self.m_robot_map[type][id].get_linear_velocity()

    def robot_set_linear_velocity(self, type, id, v):
        self.m_robot_map[type][id].set_linear_velocity(v)

    def robot_get_angular_velocity(self, type, id):
        return self.m_robot_map[type][id].get_angular_velocity()

    def robot_set_angular_velocity(self, type, id, v):
        self.m_robot_map[type][id].set_angular_velocity(v)

    def robot_get_yaw_angle(self, type, id):
        return self.m_robot_map[type][id].get_yaw_angle()

    def robot_set_yaw_angle(self, type, id, v):
        self.m_robot_map[type][id].set_yaw_angle(v)

    def robot_get_location(self, type, id):
        return self.m_robot_map[type][id].get_location()

    def robot_set_location(self, type, id, v):
        self.m_robot_map[type][id].set_location(v)

