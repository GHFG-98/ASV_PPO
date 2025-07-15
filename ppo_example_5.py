import ctypes
import numpy as np
import matplotlib.pyplot as plt
import torch
from pytorch_ppo_controller import PPOController  # 确保你已实现该类
import os
import time
import multiprocessing as mp
import copy

# --- 配置参数 ---
DLL_PATH = r'c:\研究生\03设备信息\myRIO\DLL文件 - 参数\Dll3\x64\Debug\Dll1.dll'  # DLL文件路径
TOTAL_TRAINING_STEPS = 8000000  # 总训练步数
STEPS_PER_EPISODE = 6000  # 每个episode的步数
TIME_STEP = 0.02  # 时间步长
SAVE_PLOT_EVERY_EPISODES = 50  # 每隔多少个episode保存一次图像
PLOT_DIR = "episode_plots_ppo_mp"  # 图像保存目录
RESULTS_FILENAME = 'ppo_mp_results.csv'  # 结果文件名
REWARD_PLOT_FILENAME = 'ppo_mp_rewards_plot.png'  # 奖励曲线图像文件名
MODEL_SAVE_PATH = 'ppo_mp_model.pth'  # 模型保存路径
NAN_PENALTY = -1000.0  # NaN值惩罚
XY_ANGLE_THRESHOLD = 1.0  # XY欧拉角稳定阈值（度）
XY_STABLE_REWARD = 150.0  # XY欧拉角稳定奖励基准值
ANGULAR_VEL_THRESHOLD = 50.0  # 角速度稳定阈值（度/秒）
ANGULAR_VEL_STABLE_REWARD = 120.0  # 角速度稳定奖励基准值
NUM_ENVIRONMENTS = 30  # 并行环境数量
UPDATE_TIMESTEPS = STEPS_PER_EPISODE * NUM_ENVIRONMENTS  # 更新策略的步数
# PPO 超参数
LR_ACTOR = 0.0003  # Actor学习率
LR_CRITIC = 0.001  # Critic学习率
GAMMA = 0.99  # 折扣因子
K_EPOCHS = 10  # 更新周期内的迭代次数
EPS_CLIP = 0.2  # PPO的裁剪范围
ACTION_STD_INIT = 0.1  # 初始动作标准差
MAX_GRAD_NORM = 0.5  # 最大梯度范数
# Global simulation/environment parameters (used by workers and PPO controller)
INPUT_SIZE_CONST = 6  # 动作空间维度，也是状态的一部分
OUTPUT_SIZE_CONST = 6  # DLL输出维度
EULER_DEADZONE = 360  # 欧拉角死区
ANGULAR_VELOCITY_DEADZONE = 100  # 角速度死区
MAX_OUTPUT_VALUE = 1e3  # 最大输出值
# CUDA设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
os.makedirs(PLOT_DIR, exist_ok=True)
# 设置 Matplotlib 支持中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置中文字体失败: {e}. 绘图可能无法正确显示中文。")
# 全局变量，用于主进程统计
main_process_episode_count = 0
main_process_total_steps = 0
all_episode_rewards = []  # 用于绘制平均奖励曲线

# --- 工作进程函数 ---
"""
PPO (Proximal Policy Optimization) 强化学习算法实现

本文件实现了一个基于PPO算法的控制器，用于通过DLL接口与物理仿真系统交互。
主要功能包括:
- 多进程并行训练多个环境
- 通过DLL接口调用物理仿真
- 实现PPO算法的核心训练逻辑
- 记录训练过程和结果

关键组件:
- PPOController: PPO算法实现类
- run_single_environment: 工作进程函数，负责单个环境的仿真
- 奖励函数设计: 包含欧拉角稳定、角速度稳定等奖励项

使用说明:
1. 配置DLL_PATH指向正确的DLL文件
2. 调整TOTAL_TRAINING_STEPS等超参数
3. 运行脚本开始训练

输出:
- 训练过程中的奖励曲线
- 保存的模型文件
- 每个episode的仿真结果图
----------文件保存在了：C:\研究生\03设备信息\myRIO\DLL文件 - 参数\加入强化学习6\算法文件夹\episode_plots_ppo_mp
"""
def run_single_environment(env_id, experience_queue, action_queues, state_queues, worker_args):
    dll_path = worker_args['dll_path']  # DLL文件路径
    steps_per_episode = worker_args['steps_per_episode']  # 每个episode的步数
    time_step = worker_args['time_step']  # 时间步长
    input_size = worker_args['input_size']  # 输入大小
    output_size = worker_args['output_size']  # 输出大小
    nan_penalty = worker_args['nan_penalty']  # NaN惩罚
    xy_angle_threshold = worker_args['xy_angle_threshold']  # XY角度阈值
    xy_stable_reward = worker_args['xy_stable_reward']  # XY稳定奖励
    angular_vel_threshold = worker_args['angular_vel_threshold']  # 角速度阈值
    angular_vel_stable_reward = worker_args['angular_vel_stable_reward']  # 角速度稳定奖励
    euler_deadzone = worker_args['euler_deadzone']  # 欧拉角死区
    angular_velocity_deadzone = worker_args['angular_velocity_deadzone']  # 角速度死区
    max_output_value = worker_args['max_output_value']  # 最大输出值

    # 加载DLL (每个进程独立加载)
    try:
        # 加载DLL文件并设置函数参数类型
        dll = ctypes.CDLL(dll_path)
        # 定义simulate函数参数类型: 输入数组指针, 当前时间, 时间步长, 输出数组指针
        dll.simulate.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_double,
            ctypes.c_double, ctypes.POINTER(ctypes.c_double)
        ]
        # 设置函数返回类型为None
        dll.simulate.restype = None
    except Exception as e:
        # 捕获DLL加载异常并返回
        print(f"[Worker {env_id}] 加载DLL失败: {e}")
        return

    # 打印开始运行信息
    print(f"[Worker {env_id}] 开始运行... ")

    # 初始化集数计数器
    episode_num = 0

    # 无限循环，用于处理每个集数
    while True:
        # 增加集数计数器
        episode_num += 1

        # 初始化当前输入数组，大小为(1, input_size)，数据类型为float64
        current_input_array = np.zeros((1, input_size), dtype=np.float64)

        # 初始化当前输出数组，大小为(output_size)，数据类型为float64
        current_output_array = np.zeros(output_size, dtype=np.float64)

        # 初始化前一次输出数组，大小为(output_size)，数据类型为float64
        prev_output_array_sim = np.zeros(output_size, dtype=np.float64)

        # 初始化前前一次输出数组，大小为(output_size)，数据类型为float64
        prev_prev_output_array_sim = np.zeros(output_size, dtype=np.float64)

        # 初始化当前仿真时间
        current_time_sim = 0.0

        # 初始化当前集数的欧拉角列表
        episode_euler_angles = []

        # 初始化当前集数的输入动作列表
        episode_input_actions = []

        # 📌【日志位置 1】：初始化欧拉角列表
        print(f"[Worker {env_id}] 初始化 episode_euler_angles (初始为空列表)")

        # 执行一次DLL调用获取初始非零状态
        # 调用DLL的simulate函数进行仿真
        dll.simulate(
            # 将输入数组转换为C双精度浮点数指针
            current_input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            # 当前仿真时间
            current_time_sim,
            # 时间步长
            time_step,
            # 将输出数组转换为C双精度浮点数指针
            current_output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )

        # 更新当前仿真时间
        current_time_sim += time_step

        # 保存当前输出数组作为上一个输出数组
        prev_output_array_sim = current_output_array.copy()

        # 提取初始欧拉角
        initial_euler_angles = current_output_array[3:]

        # 初始化角速度为零
        initial_angular_velocities = np.zeros(3)

        # 构建状态向量，包含输入、欧拉角和角速度
        state = np.concatenate([
            current_input_array[0], initial_euler_angles, initial_angular_velocities
        ]).flatten()

        # 记录当前集数的欧拉角
        episode_euler_angles.append(initial_euler_angles)

        # 记录当前集数的角速度
        episode_angular_velocities = []

        # 📌【日志位置 2】：首次调用 DLL 后添加初始欧拉角
        print(f"[Worker {env_id}] Step 0: 初始欧拉角 = {initial_euler_angles}")
        last_update_step = 0  # 初始化上次更新的步数
        # 遍历每个时间步
        for step in range(steps_per_episode):
            try:
                # 将状态放入队列中
                state_queues[env_id].put(state.copy())
            except Exception as e:
                # 打印发送状态失败的错误信息
                print(f"[Worker {env_id}] 发送状态失败: {e}")
                return

            try:
                # 从队列中获取动作元组，超时时间为20秒
                action_tuple = action_queues[env_id].get(timeout=20)
                # 如果动作元组为None，返回
                if action_tuple is None:
                    return
                # 解包动作和动作对数概率
                action, action_log_prob = action_tuple
            except mp.queues.Empty:
                # 打印等待动作超时的错误信息
                print(f"[Worker {env_id}] 等待动作超时，停止。")
                break
            except Exception as e:
                # 打印接收动作失败的错误信息
                print(f"[Worker {env_id}] 接收动作失败: {e}")
                break
            
            target_action = current_input_array[0].copy()  # 初始化目标动作为当前输入
                    # 对动作进行裁剪，确保其在指定范围内，并重塑为(1, -1)形状
            next_input_array_sim = np.clip(action, [0, 0, 0, -10, -10, 0], [0, 0, 0, 10, 10, 0]).reshape(1, -1)

            # 如果是第一个时间步，初始化前一个输入数组
            if step == 0:
                prev_input_array_sim = next_input_array_sim.copy()
                last_update_step = step
                target_action = next_input_array_sim[0].copy()
            else:
                # 每隔 10 步（即 0.2 秒）才允许更新一次目标动作
                if (step - last_update_step) >= 5:
                    delta = next_input_array_sim[0] - target_action
                    delta = np.clip(delta, -5.0, 5.0)  # 最大变化不超过 ±1
                    target_action += delta
                    # print(f"Worker {env_id}: last_update_step = {last_update_step}，step={step}")
                    last_update_step = step
                    # 在last_update_step = step这行代码后添加
                    
                # else:
                #     print(f"没更新差值")

            # 应用当前的目标动作
            next_input_array_sim[0] = target_action.copy()

            # 更新前一个输入数组为当前输入数组
            prev_input_array_sim = next_input_array_sim.copy()

            # 调用DLL的simulate函数进行物理仿真
            dll.simulate(
                # 将输入数组转换为C双精度浮点数指针
                current_input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                # 当前仿真时间
                current_time_sim,
                # 时间步长
                time_step,
                # 将输出数组转换为C双精度浮点数指针
                current_output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )

            # # 调试日志: 记录当前欧拉角和完整输出
            # print(f"[Worker {env_id}] Step {step}: Euler = {current_output_array[3:]}, Output = {current_output_array}")
            # episode_euler_angles2.append(copy.deepcopy(current_output_array[3:]))  # 将当前欧拉角添加到当前集数的欧拉角列表中
            # 📌
            #print(f"[Worker {env_id}] Episode {episode_num} 的欧拉角数据 (长度: {len(episode_euler_angles)}):")
            # 打印最后添加的5个欧拉角数据
            # if len(episode_euler_angles2) >= 5:
            #     print(np.array(episode_euler_angles2[-5:]))
            # else:
            #     print(np.array(episode_euler_angles2))

            # 确保以下两行在 simulate 之后执行
            # 提取当前输出数组中的欧拉角
            euler_angles_current_sim = current_output_array[3:]
            # print("1TTTTeuler_angles_current_sim: ", euler_angles_current_sim)
            # print("2TTTTprev_output_array_sim: ", prev_output_array_sim[3:] )
            # 初始化奖励为0
            reward = 0
            # 初始化当前步骤是否终止为False
            current_step_is_terminal_worker = False

            # 检查输出数组中的值是否超出最大值或包含NaN
            if np.any(np.abs(current_output_array) >= max_output_value) or np.any(np.isnan(current_output_array)):
                # 如果超出最大值或包含NaN，设置奖励为nan_penalty
                reward = nan_penalty
                # 设置当前步骤终止标志为True
                current_step_is_terminal_worker = True
            else:
                euler_angles_current_sim = current_output_array[3:]  # 提取当前模拟的欧拉角
                angular_velocities_current_sim = (euler_angles_current_sim - prev_output_array_sim[3:]) / time_step  # 计算当前模拟的角速度
                episode_euler_angles.append(copy.deepcopy(euler_angles_current_sim))  # 将当前欧拉角添加到当前集数的欧拉角列表中

                # 📌【日志位置 4】：Episode 结束时打印完整欧拉角数据
                #print(f"[Worker {env_id}] Episode {episode_num} 的欧拉角数据 (长度: {len(episode_euler_angles)}):")
                #print(np.array(episode_euler_angles))
                episode_angular_velocities.append(copy.deepcopy(angular_velocities_current_sim))  # 将当前角速度添加到当前集数的角速度列表中

                # 📌【日志位置 3】：每次调用 DLL 后打印当前欧拉角
                #print(f"[Worker {env_id}] Step {step}: Euler = {euler_angles_current_sim}")

                # 添加比较逻辑
               # if step > 0 and np.array_equal(episode_euler_angles[-1], episode_euler_angles[-2]):  # 检查当前欧拉角是否与上一步的欧拉角相同
               #     print(f"[Worker {env_id}] Warning: 欧拉角未变化！Step {step}")  # 如果相同，打印警告信息

               # 1. 欧拉角和角速度绝对值惩罚（越小越好）
                euler_penalty = np.sum(np.abs(euler_angles_current_sim))
                angular_vel_penalty = np.sum(np.abs(angular_velocities_current_sim))
                reward -= (euler_penalty + angular_vel_penalty) * 10.0

                # 2. 收敛速度奖励（数值越来越小）
                if step > 0:
                    prev_euler = episode_euler_angles[-2]
                    prev_vel = episode_angular_velocities[-2]
                    euler_improvement = np.sum(np.abs(prev_euler)) - np.sum(np.abs(euler_angles_current_sim))
                    vel_improvement = np.sum(np.abs(prev_vel)) - np.sum(np.abs(angular_velocities_current_sim))
                    reward += (euler_improvement + vel_improvement) * 50.0

                # 3. 快速收敛奖励（到小的过程越快越好）
                convergence_bonus = 1.0 / (step + 1)  # 随时间递减的奖励
                reward += convergence_bonus * 100.0

                # 4. 输入变化率惩罚（变化越慢越好）
                if step > 0:
                    input_change = np.sum(np.abs(next_input_array_sim - prev_input_array_sim))
                    reward -= input_change * 5.0

                # 保持原有的稳定性奖励
                if np.all(np.abs(euler_angles_current_sim) <= 1.0):
                    reward += XY_STABLE_REWARD
                if np.all(np.abs(angular_velocities_current_sim) <= 30.0):
                    reward += ANGULAR_VEL_STABLE_REWARD

            # 检查奖励值是否为NaN（无效值）
            if np.isnan(reward):
                # 设置惩罚奖励
                reward = nan_penalty
                # 标记当前步为终止状态
                # 如果当前步是终止步，则设置当前步为终止
                current_step_is_terminal_worker = True

            # 提取当前输出数组中的欧拉角
            euler_angles_term_check = current_output_array[3:]

            # 计算当前欧拉角与前一时刻欧拉角之间的角速度
            angular_velocities_term_check = (euler_angles_term_check - prev_output_array_sim[3:]) / time_step

            # 检查欧拉角和角速度是否超出阈值
            if (np.any(np.abs(euler_angles_term_check) > euler_deadzone) or 
                np.any(np.abs(angular_velocities_term_check) > angular_velocity_deadzone)):
                # 如果当前步不是终止步，则设置奖励为nan_penalty
                if not current_step_is_terminal_worker:
                    reward = nan_penalty
                # 设置当前步为终止
                current_step_is_terminal_worker = True

            # 判断是否完成当前集数
            done = current_step_is_terminal_worker or (step == steps_per_episode - 1)

            # 更新下一个欧拉角
            next_euler_angles = current_output_array[3:]

            # 更新下一个角速度
            next_angular_velocities = angular_velocities_current_sim

            # 构建下一个状态
            next_state = np.concatenate([
                next_input_array_sim[0], next_euler_angles, next_angular_velocities
            ]).flatten()

            # 检查下一个状态是否包含NaN值
            if np.isnan(next_state).any() and not done:
                # 如果下一个状态包含NaN值且当前步未完成，则设置奖励为nan_penalty
                reward = nan_penalty 
                # 标记当前步为完成
                done = True

            # 记录当前输入动作
            episode_input_actions.append(copy.deepcopy(next_input_array_sim[0]))

            try:
                # 如果当前步已完成
                if done:
                     # 📌【日志位置 4】：Episode 结束时打印完整欧拉角数据
                    #print(f"[Worker {env_id}] Episode {episode_num} 的欧拉角数据 (长度: {len(episode_euler_angles)}):")
                    #print(np.array(episode_euler_angles))
                    # 将当前经验数据放入经验队列
                    # 将当前经验数据放入经验队列，包括状态、动作、动作对数概率、奖励、下一个状态、是否完成、环境ID、欧拉角和角速度
                    experience_queue.put({
                        'state': state,
                        'action': action,
                        'action_log_prob': action_log_prob,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done,
                        'env_id': env_id,
                        'euler_angles': np.array(episode_euler_angles),
                        'angular_velocities': np.array(episode_angular_velocities),
                        'input_data': np.array(episode_input_actions)  # 添加 input 数据
                    })
                else:
                    # 如果当前步未完成，则将当前经验数据放入经验队列
                    experience_queue.put((state, action, action_log_prob, reward, next_state, done, env_id))
                    # 📌【日志位置 1】：Worker 发送数据前打印完整欧拉角数据
                #print(f"[Worker {env_id}] Episode {episode_num} 的欧拉角数据 (长度: {len(episode_euler_angles)}):")
                #print(np.array(episode_euler_angles))  # 打印完整数据
            except Exception as e:
                # 打印发送经验失败的错误信息
                print(f"[Worker {env_id}] 发送经验失败: {e}")
                # 退出当前循环
                break

            # 更新当前状态为下一个状态，以便在下一个时间步使用
            state = next_state.copy()  # 将下一个状态复制给当前状态

            # 更新当前输入数组为下一个输入数组，以便在下一个时间步使用
            current_input_array = next_input_array_sim.copy()  # 将下一个输入数组复制给当前输入数组

            # 更新前一个输出数组为当前输出数组，以便在下一个时间步使用
            prev_output_array_sim = current_output_array.copy()  # 将当前输出数组复制给前一个输出数组

            # 更新当前时间步
            current_time_sim += time_step  # 增加当前时间步的时间

            # 如果当前集数已完成，则跳出循环
            if done:
                break  # 结束当前集数的循环

        # 打印当前工作进程的停止信息
        print(f"[Worker {env_id}] 停止.")  # 输出当前工作进程的停止信息

if __name__ == '__main__':
    try:
        # 设置多进程的启动方法为 'spawn'，以确保子进程的独立性
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")  # 输出多进程启动方法设置成功的信息
    except RuntimeError as e:
        # 如果设置多进程启动方法时发生错误，则输出错误信息
        print(f"Note: Multiprocessing start method already set or error: {e}")  # 输出错误信息或提示多进程启动方法已设置

    # 计算状态维度，包括常量输入、欧拉角和角速度
    state_dim_calc = INPUT_SIZE_CONST + 3 + 3  # 状态维度等于常量输入维度加上欧拉角和角速度的维度

    # 计算动作维度，即常量输入的维度
    action_dim_calc = INPUT_SIZE_CONST  # 动作维度等于常量输入的维度

    # 初始化PPO控制器
    ppo_controller = PPOController(
        state_dim=state_dim_calc,  # 设置状态维度
        action_dim=action_dim_calc,  # 设置动作维度
        lr_actor=LR_ACTOR,  # 设置策略网络的学习率
        lr_critic=LR_CRITIC,  # 设置价值网络的学习率
        gamma=GAMMA,  # 设置折扣因子
        K_epochs=K_EPOCHS,  # 设置策略更新的轮数
        eps_clip=EPS_CLIP,  # 设置策略更新的剪切范围
        action_std_init=ACTION_STD_INIT,  # 设置动作标准差的初始值
        max_grad_norm=MAX_GRAD_NORM,  # 设置梯度的最大范数
        device=device  # 设置运行设备（CPU或GPU）
    )

    # 输出PPO控制器初始化完成的信息
    print(f"PPOController initialized on {device}")  # 输出控制器初始化完成的信息

    # 创建经验队列，用于存储各个工作进程的经验
    experience_queue = mp.Queue()  # 创建一个经验队列

    # 创建动作队列，用于存储各个工作进程的动作
    action_queues = [mp.Queue() for _ in range(NUM_ENVIRONMENTS)]  # 创建多个动作队列，每个工作进程一个

    # 创建状态队列，用于存储各个工作进程的状态
    state_queues = [mp.Queue() for _ in range(NUM_ENVIRONMENTS)]  # 创建多个状态队列，每个工作进程一个

    # 定义工作进程的参数
    worker_args = {
        'dll_path': DLL_PATH,  # DLL文件路径
        'steps_per_episode': STEPS_PER_EPISODE,  # 每个集数的步数
        'time_step': TIME_STEP,  # 每个时间步的时间
        'input_size': INPUT_SIZE_CONST,  # 输入数组的大小
        'output_size': OUTPUT_SIZE_CONST,  # 输出数组的大小
        'nan_penalty': NAN_PENALTY,  # NaN值的惩罚
        'xy_angle_threshold': XY_ANGLE_THRESHOLD,  # 欧拉角阈值
        'xy_stable_reward': XY_STABLE_REWARD,  # 欧拉角稳定奖励
        'angular_vel_threshold': ANGULAR_VEL_THRESHOLD,  # 角速度阈值
        'angular_vel_stable_reward': ANGULAR_VEL_STABLE_REWARD,  # 角速度稳定奖励
        'euler_deadzone': EULER_DEADZONE,  # 欧拉角死区
        'angular_velocity_deadzone': ANGULAR_VELOCITY_DEADZONE,  # 角速度死区
        'max_output_value': MAX_OUTPUT_VALUE  # 输出值的最大值
    }

    # 创建进程列表
    processes = []  # 初始化进程列表

    # 创建指定数量的工作进程
    # 遍历环境数量，创建相应数量的工作进程
    for i in range(NUM_ENVIRONMENTS):  # 遍历环境数量，创建相应数量的工作进程
        p = mp.Process(target=run_single_environment, args=(i, experience_queue, action_queues, state_queues, worker_args))  # 创建工作进程
        processes.append(p)  # 将工作进程添加到进程列表中
        p.start()  # 启动工作进程
        print(f"Worker {i} started.")  # 打印工作进程启动信息

    collected_timesteps_since_update = 0  # 初始化自上次更新以来收集的时间步数
    total_episodes_completed_main = 0  # 初始化主进程完成的总集数
    current_episode_rewards_agg = [0.0] * NUM_ENVIRONMENTS  # 初始化当前集的奖励聚合列表
    current_episode_steps_agg = [0] * NUM_ENVIRONMENTS  # 初始化当前集的时间步聚合列表

    try:
        while main_process_total_steps < TOTAL_TRAINING_STEPS:  # 当主进程总时间步数小于总训练时间步数时，继续循环
            for i in range(NUM_ENVIRONMENTS):  # 遍历环境数量
                if not state_queues[i].empty():  # 检查状态队列是否为空
                    state_np = state_queues[i].get()  # 从状态队列中获取状态
                    state_tensor = torch.FloatTensor(state_np).to(device)  # 将状态转换为张量并移动到指定设备
                    action_tensor, action_log_prob_tensor = ppo_controller.select_action(state_tensor)  # 选择动作并获取动作对数概率
                    action_np = action_tensor.detach().cpu().numpy()  # 将动作张量转换为numpy数组
                    action_log_prob_np = action_log_prob_tensor.detach().cpu().numpy()  # 将动作对数概率张量转换为numpy数组
                    action_queues[i].put((action_np, action_log_prob_np))  # 将动作和动作对数概率放入动作队列

            while not experience_queue.empty():  # 当经验队列不为空时，继续循环
                experience = experience_queue.get()  # 从经验队列中获取经验
                if isinstance(experience, dict):  # 检查经验是否为字典类型
                    state = experience['state']  # 提取状态
                    action = experience['action']  # 提取动作
                    log_prob = experience['action_log_prob']  # 提取动作对数概率
                    reward = experience['reward']  # 提取奖励
                    next_state = experience['next_state']  # 提取下一个状态
                    done = experience['done']  # 提取是否完成标志
                    worker_id = experience['env_id']  # 提取工作进程ID
                    euler_angles_data = experience['euler_angles']  # 提取欧拉角数据
                    angular_velocities_data = experience['angular_velocities']  # 提取角速度数据
                    input_data = experience['input_data']  # 提取 input 数据

                    # 📌【日志位置 5】：主进程中接收数据后打印欧拉角数据
                    # print(f"Episode {total_episodes_completed_main} 的欧拉角数据 (形状: {euler_angles_data.shape}):")
                    # print(euler_angles_data[:5])  # 打印前5行数据

                    if total_episodes_completed_main % SAVE_PLOT_EVERY_EPISODES == 0:  # 检查是否需要保存绘图
                       output_dir = os.path.join(PLOT_DIR, 'episode_outputs')  # 构建输出目录路径
                       os.makedirs(output_dir, exist_ok=True)  # 创建输出目录，如果目录已存在则不报错

                        # 保存欧拉角数据到 CSV 文件
                       euler_csv_path = os.path.join(output_dir, f'euler_angles_ep_{total_episodes_completed_main}.csv')  # 构建欧拉角数据文件路径
                       np.savetxt(euler_csv_path, euler_angles_data, delimiter=',',            
                       header='Roll,Pitch,Yaw', comments='')  # 将欧拉角数据保存到 CSV 文件

                        # 保存角速度数据到 CSV 文件
                       velocity_csv_path = os.path.join(output_dir, f'angular_velocities_ep_{total_episodes_completed_main}.csv')  # 构建角速度数据文件路径
                       np.savetxt(velocity_csv_path, angular_velocities_data, delimiter=',',            
                       header='Roll Rate,Pitch Rate,Yaw Rate', comments='')  # 将角速度数据保存到 CSV 文件

                        # 保存 input 数据到 CSV 文件
                       input_csv_path = os.path.join(output_dir, f'input_data_ep_{total_episodes_completed_main}.csv')
                       np.savetxt(input_csv_path, input_data, delimiter=',',
                                  header='Input1,Input2,Input3,Input4,Input5,Input6', comments='')

                       print(f"Episode {total_episodes_completed_main} 的数据已保存到: {output_dir}")  # 打印数据保存路径

                       # 📌【日志位置 6】：保存图像和 CSV 前打印欧拉角数据
                    # print(f"保存 Episode {total_episodes_completed_main} 的欧拉角数据 (形状: {euler_angles_data.shape}):")
                    # print(euler_angles_data[:5])
                else:
                    state, action, log_prob, reward, next_state, done, worker_id = experience  # 从经验队列中获取经验数据

                ppo_controller.buffer.add(state, action, log_prob, reward, next_state, done)  # 将经验数据添加到 PPO 控制器的缓冲区
                current_episode_rewards_agg[worker_id] += reward  # 更新当前工作进程的奖励
                current_episode_steps_agg[worker_id] += 1  # 更新当前工作进程的步数
                main_process_total_steps += 1  # 更新主进程的总步数
                collected_timesteps_since_update += 1  # 更新自上次更新以来收集的时间步数

                if done:  # 如果当前集数结束
                    total_episodes_completed_main += 1  # 更新完成的集数
                    all_episode_rewards.append(current_episode_rewards_agg[worker_id])  # 记录当前集数的奖励
                    avg_reward = np.mean(all_episode_rewards[-50:])  # 计算最近50个集数的平均奖励
                    print(f"Total Steps: {main_process_total_steps}/{TOTAL_TRAINING_STEPS}, Worker {worker_id} Episode Finished. Reward: {current_episode_rewards_agg[worker_id]:.2f}, Steps: {current_episode_steps_agg[worker_id]}, Avg Reward (last 50): {avg_reward:.2f}")  # 打印当前集数的训练信息
                    
                    # 保存平均奖励数据到CSV文件
                    avg_rewards_file = os.path.join(PLOT_DIR, "avg_rewards_all.csv")
                    with open(avg_rewards_file, 'a') as f:
                        f.write(f"{total_episodes_completed_main},{avg_reward}\n")
                    current_episode_rewards_agg[worker_id] = 0.0  # 重置当前集数的奖励
                    current_episode_steps_agg[worker_id] = 0  # 重置当前集数的步数

            if collected_timesteps_since_update >= UPDATE_TIMESTEPS:  # 如果收集的经验步数达到更新阈值
                print(f"Total Steps: {main_process_total_steps}. Updating PPO policy with {len(ppo_controller.buffer.rewards)} experiences...")  # 打印更新信息
                ppo_controller.update()  # 更新PPO策略
                collected_timesteps_since_update = 0  # 重置收集的经验步数
                print("PPO policy updated.")  # 打印更新完成信息

            if main_process_total_steps >= TOTAL_TRAINING_STEPS:  # 如果达到总训练步数
                print("达到总训练步数，停止训练。")  # 打印停止训练信息
                break  # 结束训练

            time.sleep(0.001)  # 短暂休眠，防止CPU占用过高

    except KeyboardInterrupt:  # 捕获键盘中断信号
        print("训练被用户中断。")  # 打印中断信息

    finally:  # 无论是否发生异常，执行以下代码
        print("设置停止信号，等待工作进程结束...")  # 打印进程结束信息
        for p in processes:  # 遍历所有工作进程
            p.terminate()  # 发送终止信号
            p.join()  # 等待进程结束
        print("所有工作进程已结束。")  # 打印所有进程结束信息

    plt.figure(figsize=(12, 6))  # 创建一个新的图形窗口
    plt.plot(all_episode_rewards, label='总奖励')  # 绘制所有集数的奖励曲线

    # 计算最近 50 个 episode 的平均奖励
    if len(all_episode_rewards) >= 50:  # 如果集数大于等于50
        avg_reward = np.mean(all_episode_rewards[-50:])  # 计算最近50个集数的平均奖励
    else:
        avg_reward = np.mean(all_episode_rewards)  # 否则计算所有集数的平均奖励

    # 保存平均奖励到 CSV
    reward_summary_path = os.path.join(output_dir, 'average_rewards.csv')  # 构建CSV文件路径
    with open(reward_summary_path, 'a') as f:  # 打开CSV文件，追加模式
        if os.path.getsize(reward_summary_path) == 0:  # 如果文件为空
            f.write('Episode,AvgReward50\n')  # 写入表头
        f.write(f'{total_episodes_completed_main},{avg_reward:.2f}\n')  # 写入当前集数和平均奖励

    plt.xlabel('总轮数 (所有Worker)')  # 设置x轴标签
    plt.ylabel('总奖励')  # 设置y轴标签
    plt.title('最终多进程PPO训练奖励曲线')  # 设置图表标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格

    final_reward_plot_path = os.path.join(PLOT_DIR, REWARD_PLOT_FILENAME)  # 构建奖励曲线图保存路径
    plt.savefig(final_reward_plot_path)  # 保存奖励曲线图
    plt.close()  # 关闭图表

    print(f"最终奖励曲线图已保存到: {final_reward_plot_path}")  # 打印保存路径

    final_model_path = os.path.join(PLOT_DIR, MODEL_SAVE_PATH)  # 构建模型保存路径
    ppo_controller.save(final_model_path)  # 保存最终模型
    print(f"最终模型已保存到: {final_model_path}")  # 打印保存路径

    print("训练完成。")  # 打印训练完成信息