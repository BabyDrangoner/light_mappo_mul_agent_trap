import matplotlib.pyplot as plt
import numpy as np

class plotting:
    def __init__(self):
        self.r = 0
        self.agent1_trail = []
        self.agent2_trail = []
        self.agent3_trail = []
    def initialize_plot(self, axis_range=(-10, 10)):
        """
        初始化绘图环境，设置坐标轴范围和标题。
        """
        plt.ion()  # 打开交互模式
        plt.figure(figsize=(8, 8))  # 调整窗口大小以适应两个子图
        self.agent1_trail.clear()
        self.agent2_trail.clear()
        self.agent3_trail.clear()

    def plot_agents(self, agents, targets, axis_range=(0, 20)):
        """
        绘制智能体、目标智能体的位置。
        agents 和 targets 都是两行三列的矩阵，其中每列代表一个智能体/目标的x和y坐标。
        """
        # 清除当前图形
        plt.clf()

        plt.axis('square')

        colors = ['#377eb8', '#4daf4a', '#e41a1c']  # 深蓝色，墨绿色，暗红色

        self.agent1_trail.append((agents[0, 0], agents[1, 0]))
        self.agent2_trail.append((agents[0, 1], agents[1, 1]))
        self.agent3_trail.append((agents[0, 2], agents[1, 2]))

        # 绘制智能体位置
        # agents[0, :] 是所有智能体的x坐标，agents[1, :] 是所有智能体的y坐标
        for i in range(3):
            plt.plot(agents[0, i], agents[1, i], color=colors[i], marker='o', label='agent_{}'.format(i))

        # 绘制目标位置
        # targets[0, :] 是所有目标的x坐标，targets[1, :] 是所有目标的y坐标
        plt.plot(targets[0, :], targets[1, :], 'bo',label='target_0')  # 'bo'代表蓝色圆点

        for agent_trail, color in zip([self.agent1_trail, self.agent2_trail, self.agent3_trail], colors):
            x_vals, y_vals = zip(*agent_trail)  # 解包轨迹坐标
            plt.plot(x_vals, y_vals, color=color, linestyle='--')  # 绘制轨迹

        # 设置坐标轴范围
        plt.xlim(axis_range[0], axis_range[1])
        plt.ylim(axis_range[0], axis_range[1])

        # 添加水平和垂直的参考线
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)

        # 显示网格
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        # 设置图表标题

        # 更新绘图
        plt.draw()
        plt.pause(0.0001)  # 暂停一段时间，然后继续执行代码

    import matplotlib.pyplot as plt

    def plot_simulation_data(self, theta_dict, distance_dict):
        """
        Plot simulation data for theta and distance dictionaries.

        Parameters:
        - theta_dict: Dictionary with keys '0', '1', '2' and list values containing theta values for agents
        - distance_dict: Dictionary with keys '0', '1', '2' and list values containing distance values for agents
        """
        fig, axes = plt.subplots(1, 2, figsize=(4, 2))  # 创建两行一列的图窗

        # 绘制 theta 数据
        for key, values in theta_dict.items():
            axes[0].plot(values, label=f'Agent {key} Theta')
        axes[0].set_title('Theta Values by Agent')
        axes[0].legend()
        axes[0].set_ylabel('Theta')

        # 绘制 distance 数据
        for key, values in distance_dict.items():
            axes[1].plot(values, label=f'Agent {key} Distance')
        axes[1].set_title('Distance Values by Agent')
        axes[1].legend()
        axes[1].set_ylabel('Distance')
        plt.grid()
        plt.tight_layout()
        plt.show()



    def close_plot(self):
        """
        关闭当前的绘图窗口。
        """
        plt.close()