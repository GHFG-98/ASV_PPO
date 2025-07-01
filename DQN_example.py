import ctypes
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
from pytorch_dqn_controller import DQNController
import os
import time
import multiprocessing as mp

# --- 配置参数 ---
DLL_PATH = r'c:\研究生\03设备信息\myRIO\DLL文件 - 参数\Dll1\x64\Debug\Dll1.dll'
NUM_ENVIRONMENTS = 10 # 并行环境数量
NUM_EPISODES_PER_WORKER = 50 # 每个worker大致负责的轮数 (总轮数 = NUM_ENVIRONMENTS * NUM_EPISODES_PER_WORKER)
STEPS_PER_EPISODE = 500
TIME_STEP = 0.01
SAVE_PLOT_EVERY = 10 # 按总轮数计算 (修改为10轮)
PLOT_DIR = "episode_plots"
RESULTS_FILENAME = 'simulation_results_final.csv'
OUTPUT_PLOT_FILENAME = 'output3_all_episodes_plot.png'
REWARD_PLOT_FILENAME = 'episode_rewards_plot.png'

BATCH_SIZE = 32 # 定义批量大小常量

# --- 工作进程函数 ---
def run_single_environment(env_id, experience_queue, action_queues, state_queues, stop_event, worker_args):
    """每个工作进程运行的函数"""
    num_episodes = worker_args['num_episodes_per_worker']
    steps_per_episode = worker_args['steps_per_episode']
    time_step = worker_args['time_step']
    input_size = worker_args['input_size']
    output_size = worker_args['output_size']
    save_plot_every = worker_args['save_plot_every'] # 获取保存间隔
    plot_dir = worker_args['plot_dir'] # 获取绘图目录

    # 加载DLL (每个进程独立加载)
    try:
        dll = ctypes.CDLL(DLL_PATH)
        dll.simulate.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_double,
            ctypes.c_double, ctypes.POINTER(ctypes.c_double)
        ]
        dll.simulate.restype = None
    except Exception as e:
        print(f"[Worker {env_id}] 加载DLL失败: {e}")
        return

    print(f"[Worker {env_id}] 开始运行 {num_episodes} 轮...")

    for episode in range(num_episodes):
        if stop_event.is_set():
            print(f"[Worker {env_id}] 收到停止信号，退出。")
            break

        input_array = np.zeros((1, input_size), dtype=np.float64)
        output_array = np.zeros(output_size, dtype=np.float64)
        prev_output_array = np.zeros(output_size, dtype=np.float64) # 存储上一步的输出
        current_time = 0.0
        episode_reward = 0
        # episode_results = [] # 不再需要完整结果列表
        episode_output3_data = [] # 用于存储 (time, output[3])
        episode_euler_data = [] # 用于存储欧拉角 (time, roll, pitch, yaw)
        episode_angular_velocity_data = [] # 用于存储角速度 (time, wx, wy, wz)
        change_penalty_weight = 0.1 # 新增：变化惩罚权重，可以调整

        for step in range(steps_per_episode):
            if stop_event.is_set(): break

            # 1. 计算当前状态
            current_output_reshaped = output_array.reshape(1, -1)
            state = np.concatenate([input_array, current_output_reshaped], axis=1)

            # 调试：检查计算出的状态是否包含 NaN
            if np.isnan(state).any():
                print(f"!!! [Worker {env_id}] Step {step+1}: NaN detected in calculated state before sending: {state.flatten()} !!!")
                # 这通常意味着上一步的 output_array 或 input_array 已经是 NaN
                # stop_event.set()
                # break

            # 2. 将状态发送给主进程，请求动作
            try:
                state_queues[env_id].put((env_id, state))
            except Exception as e:
                print(f"[Worker {env_id}] 发送状态失败: {e}")
                stop_event.set()
                break

            # 3. 从主进程接收动作
            try:
                action = action_queues[env_id].get(timeout=10) # 等待动作
                if action is None: # 主进程可能要求停止
                    stop_event.set()
                    break
                action = np.array(action, dtype=np.float64).reshape(1, -1)
            except mp.queues.Empty:
                print(f"[Worker {env_id}] 等待动作超时，停止。")
                stop_event.set()
                break
            except Exception as e:
                 print(f"[Worker {env_id}] 接收动作失败: {e}")
                 stop_event.set()
                 break

            # 调试：检查接收到的动作是否包含 NaN
            if np.isnan(action).any():
                print(f"!!! [Worker {env_id}] Step {step+1}: NaN detected in received action: {action.flatten()} !!!")
                # 可以选择停止或跳过此步骤
                # stop_event.set()
                # break

            # 调试：打印接收到的动作和即将传递给DLL的输入
            # print(f"[Worker {env_id}] Step {step+1} Received Action: {action.flatten()}")

            # 4. 执行仿真
            # --- 添加输入裁剪 (根据要求分别限制范围) ---
            original_action_flat = action.flatten()
            clipped_action = np.zeros_like(action)
            # 限制前三个参数在 [-20, 20]
            clipped_action[0, :3] = np.clip(action[0, :3], -20.0, 20.0)
            # 限制后三个参数在 [-5, 5]
            clipped_action[0, 3:] = np.clip(action[0, 3:], -5.0, 5.0)

            clipped_action_flat = clipped_action.flatten()
            # if not np.array_equal(original_action_flat, clipped_action_flat):
            #     print(f"[Worker {env_id}] Step {step+1} Action clipped from {original_action_flat} to {clipped_action_flat}")
            next_input_array = clipped_action
            # --- 输入裁剪结束 ---
            # print(f"[Worker {env_id}] Step {step+1} Input to DLL (clipped): {next_input_array.flatten()}")
            dll.simulate(
                next_input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                current_time,
                time_step,
                output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            )
            # 调试：打印 DLL 输出
            # print(f"[Worker {env_id}] Step {step+1} DLL Output: {output_array}")
            if np.isnan(output_array).any():
                 print(f"!!! [Worker {env_id}] Step {step+1}: NaN detected in DLL output_array: {output_array} !!!")

            # 5. 计算奖励和下一个状态
            # 基础奖励：惩罚输出的绝对值
            base_reward = -np.sum(np.abs(output_array))
            # 变化惩罚：惩罚后三个参数的变化率
            output_change = output_array[3:] - prev_output_array[3:]
            change_penalty = -change_penalty_weight * np.sum(np.abs(output_change))
            reward = base_reward + change_penalty

            # 调试：打印奖励
            # print(f"[Worker {env_id}] Step {step+1} Base Reward: {base_reward:.4f}, Change Penalty: {change_penalty:.4f}, Total Reward: {reward:.4f}")
            if np.isnan(reward):
                print(f"!!! [Worker {env_id}] Step {step+1}: NaN detected in calculated reward (Base: {base_reward}, Change Penalty: {change_penalty}) !!!")

            episode_reward += reward
            next_output_reshaped = output_array.reshape(1, -1)
            next_state = np.concatenate([next_input_array, next_output_reshaped], axis=1)

            # 6. 检查终止条件
            done = (step == steps_per_episode - 1)
            early_termination = False
            if np.all(output_array[3:] == 90):
                print(f"[Worker {env_id}] 轮次 {episode+1}, 步骤 {step+1}: 检测到终止条件，提前结束本轮。")
                done = True
                early_termination = True

            # 7. 将经验发送给主进程
            try:
                experience = (state, action.flatten(), reward, next_state, done)
                experience_queue.put(experience)
            except Exception as e:
                print(f"[Worker {env_id}] 发送经验失败: {e}")
                stop_event.set()
                break

            # 记录数据 (可选，如果需要每个worker保存)
            # episode_results.append([current_time + time_step] + list(output_array))
            # 记录数据
            current_time_point = current_time + time_step
            episode_output3_data.append((current_time_point, output_array[3])) # 记录 output[3]
            episode_euler_data.append((current_time_point, output_array[0], output_array[1], output_array[2])) # 记录欧拉角
            episode_angular_velocity_data.append((current_time_point, output_array[3], output_array[4], output_array[5])) # 记录角速度
            # 保存当前数据到CSV文件
            csv_filename = os.path.join('state_data', f'state_worker_{env_id}_episode_{episode+1}.csv')
            try:
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if step == 0:  # 如果是第一步，写入表头
                        writer.writerow(['Time', 'Roll', 'Pitch', 'Yaw', 'AngVel_X', 'AngVel_Y', 'AngVel_Z'])
                    writer.writerow([current_time_point] + [output_array[i] for i in range(6)])
            except PermissionError as e:
                print(f"!!! [Worker {env_id}] Episode {episode+1}, Step {step+1}: Permission denied when writing to {csv_filename}. Error: {e} !!!")
                # 可以选择停止或记录错误并继续
            except Exception as e:
                print(f"!!! [Worker {env_id}] Episode {episode+1}, Step {step+1}: Error writing to {csv_filename}. Error: {e} !!!")

            # 更新状态和时间
            input_array = next_input_array
            prev_output_array = output_array.copy() # 更新上一步的输出
            current_time += time_step

            if early_termination:
                break # 结束当前轮

        # print(f"[Worker {env_id}] 轮次 {episode+1} 结束, 总奖励: {episode_reward:.4f}")
        # 可以将每轮奖励也放入队列
        try:
            experience_queue.put((None, None, episode_reward, None, None, env_id)) # 特殊标记表示轮次结束和奖励
        except Exception as e:
            print(f"[Worker {env_id}] 发送轮次奖励失败: {e}")

        # 在指定间隔绘制并保存 output[3] 图像
        if (episode + 1) % save_plot_every == 0:
            print(f"[Worker {env_id}] Episode {episode+1}: Attempting to plot data. Output[3] data points: {len(episode_output3_data)}, Euler data points: {len(episode_euler_data)}, Angular velocity data points: {len(episode_angular_velocity_data)}") # 取消注释
            try:
                # --- Plotting Output[3] ---
                if episode_output3_data:
                    times_o3 = [data[0] for data in episode_output3_data]
                    output3_values = [data[1] for data in episode_output3_data]
                    has_nan_o3 = np.isnan(output3_values).any()
                    print(f"[Worker {env_id}] Episode {episode+1}: Checking for NaN in Output[3] plot data. Has NaN: {has_nan_o3}")

                    times_to_plot_o3 = times_o3
                    output3_to_plot = output3_values
                    if has_nan_o3:
                        print(f"!!! [Worker {env_id}] Episode {episode+1}: NaN detected in output3_values for plotting !!!")
                        valid_indices_o3 = ~np.isnan(output3_values)
                        times_np_o3 = np.array(times_o3)
                        output3_values_np = np.array(output3_values)
                        times_filtered_o3 = times_np_o3[valid_indices_o3]
                        output3_values_filtered = output3_values_np[valid_indices_o3]
                        print(f"[Worker {env_id}] Episode {episode+1}: Filtered NaN for Output[3]. Valid data points: {len(times_filtered_o3)}")
                        if len(times_filtered_o3) == 0:
                            print(f"[Worker {env_id}] Episode {episode+1}: No valid Output[3] data left after filtering NaN.")
                        else:
                            times_to_plot_o3 = times_filtered_o3
                            output3_to_plot = output3_values_filtered
                    
                    if len(times_to_plot_o3) > 0:
                        plt.figure(figsize=(10, 6))
                        plt.plot(times_to_plot_o3, output3_to_plot, 'g-', linewidth=1)
                        plt.title(f'Worker {env_id} - 轮次 {episode+1} - Output[3] 变化')
                        plt.xlabel('时间 (s)')
                        plt.ylabel('Output[3]')
                        plt.grid(True)
                        plot_filename_o3 = os.path.join(plot_dir, f'output3_worker_{env_id}_episode_{episode+1}.png')
                        print(f"[Worker {env_id}] Episode {episode+1}: Saving Output[3] plot to {plot_filename_o3}")
                        plt.savefig(plot_filename_o3)
                        plt.close() # Close the figure
                        print(f"[Worker {env_id}] Episode {episode+1}: Output[3] plot saved.")
                    else:
                        print(f"[Worker {env_id}] Episode {episode+1}: No data to plot for Output[3].")
                else:
                    print(f"[Worker {env_id}] Episode {episode+1}: No Output[3] data to plot.")

                # --- Plotting Euler Angles ---
                if episode_euler_data:
                    euler_times = [data[0] for data in episode_euler_data]
                    roll_values = [data[1] for data in episode_euler_data]
                    pitch_values = [data[2] for data in episode_euler_data]
                    yaw_values = [data[3] for data in episode_euler_data]

                    # Basic NaN check for Euler angles (can be enhanced like Output[3] if needed)
                    if np.isnan(roll_values).any() or np.isnan(pitch_values).any() or np.isnan(yaw_values).any():
                        print(f"!!! [Worker {env_id}] Episode {episode+1}: NaN detected in Euler angle data for plotting. Plotting valid points only. !!!")
                        # A simple filter: plot only if all components for a point are valid (can be done per series too)
                        valid_euler_indices = ~ (np.isnan(roll_values) | np.isnan(pitch_values) | np.isnan(yaw_values))
                        euler_times_np = np.array(euler_times)

                        euler_times_plot = euler_times_np[valid_euler_indices]
                        roll_values_plot = np.array(roll_values)[valid_euler_indices]
                        pitch_values_plot = np.array(pitch_values)[valid_euler_indices]
                        yaw_values_plot = np.array(yaw_values)[valid_euler_indices]
                        if len(euler_times_plot) == 0:
                             print(f"[Worker {env_id}] Episode {episode+1}: No valid Euler data after filtering NaN.")
                        else:
                            print(f"[Worker {env_id}] Episode {episode+1}: Filtered NaN for Euler angles. Valid data points: {len(euler_times_plot)}")
                    else:
                        euler_times_plot = euler_times
                        roll_values_plot = roll_values
                        pitch_values_plot = pitch_values
                        yaw_values_plot = yaw_values
                    
                    if len(euler_times_plot) > 0:
                        plt.figure(figsize=(10, 6))
                        plt.plot(euler_times_plot, roll_values_plot, 'r-', label='Roll', linewidth=1)
                        plt.plot(euler_times_plot, pitch_values_plot, 'g-', label='Pitch', linewidth=1)
                        plt.plot(euler_times_plot, yaw_values_plot, 'b-', label='Yaw', linewidth=1)
                        plt.title(f'Worker {env_id} - 轮次 {episode+1} - 欧拉角变化')
                        plt.xlabel('时间 (s)')
                        plt.ylabel('角度 (deg)')
                        plt.grid(True)
                        plt.legend()
                        plot_filename_euler = os.path.join(plot_dir, f'euler_angles_worker_{env_id}_episode_{episode+1}.png')
                        print(f"[Worker {env_id}] Episode {episode+1}: Saving Euler angles plot to {plot_filename_euler}")
                        plt.savefig(plot_filename_euler)
                        plt.close() # Close the figure
                        print(f"[Worker {env_id}] Episode {episode+1}: Euler angles plot saved.")
                    else:
                        print(f"[Worker {env_id}] Episode {episode+1}: No data to plot for Euler angles.")
                else:
                    print(f"[Worker {env_id}] Episode {episode+1}: No Euler angle data to plot.")

                # --- Plotting Angular Velocities ---
                if episode_angular_velocity_data:
                    angular_vel_times = [data[0] for data in episode_angular_velocity_data]
                    wx_values = [data[1] for data in episode_angular_velocity_data]
                    wy_values = [data[2] for data in episode_angular_velocity_data]
                    wz_values = [data[3] for data in episode_angular_velocity_data]

                    # Basic NaN check for angular velocities (can be enhanced like Output[3] if needed)
                    if np.isnan(wx_values).any() or np.isnan(wy_values).any() or np.isnan(wz_values).any():
                        print(f"!!! [Worker {env_id}] Episode {episode+1}: NaN detected in angular velocity data for plotting. Plotting valid points only. !!!")
                        valid_ang_vel_indices = ~ (np.isnan(wx_values) | np.isnan(wy_values) | np.isnan(wz_values))
                        angular_vel_times_np = np.array(angular_vel_times)

                        angular_vel_times_plot = angular_vel_times_np[valid_ang_vel_indices]
                        wx_values_plot = np.array(wx_values)[valid_ang_vel_indices]
                        wy_values_plot = np.array(wy_values)[valid_ang_vel_indices]
                        wz_values_plot = np.array(wz_values)[valid_ang_vel_indices]
                        if len(angular_vel_times_plot) == 0:
                            print(f"[Worker {env_id}] Episode {episode+1}: No valid angular velocity data after filtering NaN.")
                        else:
                            print(f"[Worker {env_id}] Episode {episode+1}: Filtered NaN for angular velocities. Valid data points: {len(angular_vel_times_plot)}")
                    else:
                        angular_vel_times_plot = angular_vel_times
                        wx_values_plot = wx_values
                        wy_values_plot = wy_values
                        wz_values_plot = wz_values

                    if len(angular_vel_times_plot) > 0:
                        plt.figure(figsize=(10, 6))
                        plt.plot(angular_vel_times_plot, wx_values_plot, 'r-', label='ωx', linewidth=1)
                        plt.plot(angular_vel_times_plot, wy_values_plot, 'g-', label='ωy', linewidth=1)
                        plt.plot(angular_vel_times_plot, wz_values_plot, 'b-', label='ωz', linewidth=1)
                        plt.title(f'Worker {env_id} - 轮次 {episode+1} - 角速度变化')
                        plt.xlabel('时间 (s)')
                        plt.ylabel('角速度 (rad/s)')
                        plt.grid(True)
                        plt.legend()
                        plot_filename_ang_vel = os.path.join(plot_dir, f'angular_velocity_worker_{env_id}_episode_{episode+1}.png')
                        print(f"[Worker {env_id}] Episode {episode+1}: Saving angular velocity plot to {plot_filename_ang_vel}")
                        plt.savefig(plot_filename_ang_vel)
                        plt.close() # Close the figure
                        print(f"[Worker {env_id}] Episode {episode+1}: Angular velocity plot saved.")
                    else:
                        print(f"[Worker {env_id}] Episode {episode+1}: No data to plot for angular velocities.")
                else:
                    print(f"[Worker {env_id}] Episode {episode+1}: No angular velocity data to plot.")
                
            except Exception as e:
                print(f"[Worker {env_id}] Episode {episode+1}: Error during plotting: {e}")

    print(f"[Worker {env_id}] 完成所有轮次。")

# --- 主进程 --- 
def main():
    # 设置 Matplotlib 支持中文显示
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    except Exception as e:
        print(f"[Main] 设置中文字体失败: {e}. 绘图可能无法正确显示中文。")

    # 定义目录常量
    STATE_DATA_DIR = 'state_data'

    os.makedirs(PLOT_DIR, exist_ok=True) # PLOT_DIR is "episode_plots"

    # 创建 state_data 目录并检查可写权限
    try:
        # 尝试创建目录，如果已存在则什么也不做 (exist_ok=True)
        os.makedirs(STATE_DATA_DIR, exist_ok=True)
        print(f"[Main] 已确保目录存在: {STATE_DATA_DIR}")

        # 测试写入权限：尝试在目录中创建并删除一个临时文件
        # 这有助于在启动工作进程前发现权限问题
        test_file_path = os.path.join(STATE_DATA_DIR, ".permission_test.tmp")
        with open(test_file_path, 'w') as temp_file:
            temp_file.write("test_write_permissions")
        os.remove(test_file_path) # 清理临时文件
        print(f"[Main] 目录 {STATE_DATA_DIR} 可写。")

    except OSError as e:
        print(f"[Main] ########## 关键错误 ##########")
        print(f"[Main] 无法创建或写入数据目录 '{STATE_DATA_DIR}'. 程序无法继续。")
        print(f"[Main] 请检查以下可能的原因:")
        print(f"[Main]   1. 脚本运行权限: Python脚本是否具有在当前工作目录下 '{os.getcwd()}' 创建和写入 '{STATE_DATA_DIR}' 的权限?")
        print(f"[Main]   2. 磁盘空间: 磁盘是否已满?")
        print(f"[Main]   3. 文件/目录冲突: 是否存在一个名为 '{STATE_DATA_DIR}' 的文件 (而不是目录) 阻止了目录操作?")
        print(f"[Main]   4. 杀毒软件或安全策略: 是否有安全软件阻止了文件写入操作?")
        print(f"[Main] 操作系统返回的详细错误: {e}")
        print(f"[Main] ##############################")
        return # 关键目录操作失败，退出程序

    # 共享资源
    experience_queue = mp.Queue()
    action_queues = [mp.Queue() for _ in range(NUM_ENVIRONMENTS)]
    state_queues = [mp.Queue() for _ in range(NUM_ENVIRONMENTS)]
    stop_event = mp.Event()

    # 初始化DQN控制器 (在主进程)
    # 假设 DQNController 的输入状态维度是 input_size + output_size，动作维度是 input_size
    # 需要知道确切的维度
    input_size = 6 # 假设输入维度为6
    output_size = 6 # 假设输出维度为6
    state_dim = input_size + output_size
    action_dim = input_size
    # dqn_controller = DQNController(state_dim=state_dim, action_dim=action_dim) # 需要传递维度
    dqn_controller = DQNController(state_size=state_dim, action_size=action_dim) # 使用正确的参数名

    worker_args = {
        'num_episodes_per_worker': NUM_EPISODES_PER_WORKER,
        'steps_per_episode': STEPS_PER_EPISODE,
        'time_step': TIME_STEP,
        'input_size': input_size,
        'output_size': output_size,
        'save_plot_every': SAVE_PLOT_EVERY, # 传递保存间隔
        'plot_dir': PLOT_DIR # 传递绘图目录
    }

    # 创建并启动工作进程
    workers = []
    for i in range(NUM_ENVIRONMENTS):
        p = mp.Process(target=run_single_environment, args=(i, experience_queue, action_queues, state_queues, stop_event, worker_args))
        workers.append(p)
        p.start()

    print(f"启动了 {NUM_ENVIRONMENTS} 个工作进程...")

    total_steps = 0
    total_episodes = 0
    all_episode_rewards = []
    # all_results_for_plotting = [] # 收集绘图数据可能需要更复杂的处理

    active_workers = NUM_ENVIRONMENTS
    last_plot_episode = 0

    try:
        while active_workers > 0:
            # 1. 处理来自工作进程的状态请求，生成并发送动作
            pending_actions = 0
            for i in range(NUM_ENVIRONMENTS):
                try:
                    env_id, state = state_queues[i].get_nowait()
                    action = dqn_controller.act(state) # 主进程生成动作
                    action_queues[env_id].put(action)
                    pending_actions += 1
                except mp.queues.Empty:
                    pass # 没有来自这个worker的状态请求
                except Exception as e:
                    print(f"[Main] 处理来自 Worker {i} 的状态时出错: {e}")
                    stop_event.set()

            # 2. 从经验队列收集经验并训练
            collected_experiences = 0
            while not experience_queue.empty():
                try:
                    exp_data = experience_queue.get_nowait()
                    # 检查是否是轮次结束标记
                    if len(exp_data) == 6 and exp_data[0] is None:
                        _, _, ep_reward, _, _, worker_id = exp_data
                        all_episode_rewards.append(ep_reward)
                        total_episodes += 1
                        print(f"[Main] Worker {worker_id} 完成轮次 {total_episodes}/{NUM_ENVIRONMENTS * NUM_EPISODES_PER_WORKER}, 奖励: {ep_reward:.4f}")
                        # 检查是否需要绘图 (绘制总奖励曲线)
                        if total_episodes % SAVE_PLOT_EVERY == 0 and total_episodes > last_plot_episode:
                             # 绘制并保存当前所有轮次的总奖励变化曲线
                             try:
                                 plt.figure(figsize=(10, 6))
                                 # 平滑处理奖励曲线可能更好看
                                 window_size = 50
                                 # 调试：检查总奖励数据
                                 print(f"[Main] Plotting total rewards up to episode {total_episodes}. Data points: {len(all_episode_rewards)}")
                                 valid_rewards = [r for r in all_episode_rewards if not np.isnan(r)]
                                 print(f"[Main] Valid reward data points: {len(valid_rewards)}")

                                 if not valid_rewards:
                                     print("[Main] No valid reward data to plot.")
                                     plt.close()
                                     continue

                                 # 绘制有效奖励
                                 if len(valid_rewards) >= window_size:
                                     # 注意：平滑处理基于有效奖励，x轴需要对应调整
                                     smoothed_rewards = np.convolve(valid_rewards, np.ones(window_size)/window_size, mode='valid')
                                     # 绘制平滑曲线，x轴需要计算
                                     # 假设原始索引是基于 all_episode_rewards 的，需要映射
                                     original_indices = [i for i, r in enumerate(all_episode_rewards) if not np.isnan(r)]
                                     smoothed_x = original_indices[window_size-1:] # 近似处理，可能不完全精确
                                     if len(smoothed_x) == len(smoothed_rewards):
                                         plt.plot(smoothed_x, smoothed_rewards, 'r-', linewidth=1, label=f'Smoothed (window={window_size})')
                                     else:
                                         print("[Main] Warning: Smoothed reward plotting x-axis mismatch.")
                                         # Fallback: plot smoothed rewards against simple range
                                         plt.plot(range(window_size -1, len(valid_rewards)), smoothed_rewards, 'r-', linewidth=1, label=f'Smoothed (window={window_size}, approx. x-axis)')

                                     plt.plot(original_indices, valid_rewards, 'b-', alpha=0.3, linewidth=0.5, label='Raw Valid Rewards')
                                 else:
                                     # If not enough data for smoothing, plot raw valid rewards
                                     original_indices = [i for i, r in enumerate(all_episode_rewards) if not np.isnan(r)]
                                     plt.plot(original_indices, valid_rewards, 'r-', linewidth=1, label='Raw Valid Rewards')
                                     # Removed the redundant else block here

                                 plt.title(f'每轮总奖励变化曲线 (截至第 {total_episodes} 轮)')
                                 plt.xlabel('轮次')
                                 plt.ylabel('总奖励')
                                 plt.grid(True)
                                 plt.legend()
                                 plot_filename = os.path.join(PLOT_DIR, f'reward_plot_episode_{total_episodes}.png')
                                 plt.savefig(plot_filename)
                                 print(f"[Main] 已生成奖励曲线图，保存为 {plot_filename}")
                                 plt.close()
                             except Exception as e:
                                 print(f"[Main] 绘制奖励曲线时出错: {e}")
                             last_plot_episode = total_episodes
                    elif len(exp_data) == 5:
                        state, action, reward, next_state, done = exp_data
                        dqn_controller.remember(state, action, reward, next_state, done)
                        total_steps += 1
                        collected_experiences += 1
                        # 控制训练频率
                        # if total_steps % dqn_controller.batch_size == 0 and len(dqn_controller.memory) > dqn_controller.batch_size:
                        if total_steps % BATCH_SIZE == 0 and len(dqn_controller.memory) > BATCH_SIZE:
                             # dqn_controller.replay(batch_size=BATCH_SIZE) # 传递 batch_size 给 replay 方法
                             dqn_controller.replay() # 使用方法内部的默认 batch_size
                             # print(f"[Main] Step {total_steps}: Replay finished.")
                    else:
                         print(f"[Main] 从队列收到未知数据: {exp_data}")

                except mp.queues.Empty:
                    break # 队列暂时为空
                except Exception as e:
                    print(f"[Main] 处理经验时出错: {e}")
                    stop_event.set()

            # 检查是否有进程结束
            new_active_workers = 0
            for p in workers:
                if p.is_alive():
                    new_active_workers += 1
            if new_active_workers < active_workers:
                 print(f"[Main] 检测到有 {active_workers - new_active_workers} 个工作进程结束。剩余 {new_active_workers} 个。")
            active_workers = new_active_workers

            # 如果没有待处理的动作且没有收集到经验，稍微等待一下避免CPU空转
            if pending_actions == 0 and collected_experiences == 0 and active_workers > 0:
                time.sleep(0.01)

            # 检查是否所有预定轮次都已报告完成
            if total_episodes >= NUM_ENVIRONMENTS * NUM_EPISODES_PER_WORKER:
                print("[Main] 所有预定轮次已完成，准备停止。")
                stop_event.set()

    except KeyboardInterrupt:
        print("[Main] 检测到 Ctrl+C，正在停止所有工作进程...")
        stop_event.set()

    finally:
        print("[Main] 等待所有工作进程终止...")
        # 确保队列为空，避免进程阻塞
        for q in action_queues + state_queues:
            while not q.empty():
                try: q.get_nowait()
                except mp.queues.Empty: break
            q.close()
            q.join_thread()
        while not experience_queue.empty():
             try: experience_queue.get_nowait()
             except mp.queues.Empty: break
        experience_queue.close()
        experience_queue.join_thread()

        for p in workers:
            p.join(timeout=5)
            if p.is_alive():
                print(f"[Main] Worker {p.pid} 未能在5秒内终止，强制终止。")
                p.terminate()
                p.join()
        print("[Main] 所有工作进程已终止。")

    print(f"\n训练完成! 总步数: {total_steps}, 总轮数: {total_episodes}")

    # --- 结果处理和绘图 (需要调整) ---
    # 保存最终模型
    # dqn_controller.save_model('final_dqn_model.pth') # 假设有保存模型的方法

    # 绘制每轮总奖励的变化
    if all_episode_rewards:
        try:
            plt.figure(figsize=(10, 6))
            # 平滑处理奖励曲线可能更好看
            window_size = 50
            if len(all_episode_rewards) >= window_size:
                smoothed_rewards = np.convolve(all_episode_rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(range(window_size -1, len(all_episode_rewards)), smoothed_rewards, 'r-', linewidth=1, label=f'Smoothed (window={window_size})')
                plt.plot(range(len(all_episode_rewards)), all_episode_rewards, 'b-', alpha=0.3, linewidth=0.5, label='Raw')
            else:
                 plt.plot(range(len(all_episode_rewards)), all_episode_rewards, 'r-', linewidth=1, label='Raw')

            plt.title('每轮总奖励变化曲线')
            plt.xlabel('轮次')
            plt.ylabel('总奖励')
            plt.grid(True)
            plt.legend()
            plt.savefig(REWARD_PLOT_FILENAME)
            print(f"已生成每轮总奖励变化曲线图，保存为 {REWARD_PLOT_FILENAME}")
            plt.close()
        except Exception as e:
            print(f"绘制奖励曲线时出错: {e}")
    else:
        print("没有收集到足够的奖励数据用于绘图。")

    # 注意：保存详细仿真结果和绘制output[3]曲线在此并行结构下变得复杂，
    # 因为数据分散在各个进程中，并且经验队列只传输了训练所需的核心数据。
    # 如果需要这些图，需要在worker进程中实现数据记录和保存，或者设计更复杂的
    # 数据回传机制。
    print(f"注意：详细的 {RESULTS_FILENAME} 和全局 {OUTPUT_PLOT_FILENAME} 在此并行版本中未实现。")
    # print(f"定期保存的总奖励曲线图保存在 '{PLOT_DIR}' 目录中。")
    print(f"定期保存的总奖励曲线图和各 Worker 的 Output[3] 曲线图保存在 '{PLOT_DIR}' 目录中。")

if __name__ == "__main__":
    # 在Windows上使用 'spawn' 启动方式通常更稳定
    # mp.set_start_method('spawn') # 取消注释如果遇到问题
    main()