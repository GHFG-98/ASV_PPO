# 导入必要的库
import torch  # PyTorch深度学习框架
import numpy as np  # 数值计算库
from pytorch_ppo_controller import PPOController  # PPO控制器类
import matplotlib.pyplot as plt  # 绘图库
import ctypes  # 用于调用DLL
import time  # 时间相关功能

# 加载训练好的模型路径
MODEL_PATH = r'C:\研究生\03设备信息\myRIO\DLL文件 - 参数\episode_plots_ppo_mp\ppo_mp_model.pth'

# 初始化PPO控制器参数
state_dim = 12  # 状态维度
action_dim = 6  # 动作维度
TEST_DURATION_SECONDS = 50  # 测试总时长(秒)
TIME_STEP = 0.01  # 时间步长

# 设置matplotlib中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建PPO控制器实例
ppo_controller = PPOController(
    state_dim=state_dim,  # 状态维度
    action_dim=action_dim,  # 动作维度
    lr_actor=0.0003,  # Actor网络学习率
    lr_critic=0.001,  # Critic网络学习率
    gamma=0.99,  # 折扣因子
    K_epochs=10,  # 更新策略时的epoch数
    eps_clip=0.2,  # PPO的clip参数
    action_std_init=0.1,  # 动作标准差初始值
    max_grad_norm=0.5,  # 梯度裁剪阈值
    device='cuda' if torch.cuda.is_available() else 'cpu'  # 使用GPU或CPU
)

# 加载预训练模型
ppo_controller.load(MODEL_PATH)
print(f"成功加载模型: {MODEL_PATH}")

# 计算测试总步数
test_steps = int(TEST_DURATION_SECONDS / TIME_STEP)

# 初始化历史数据存储列表
state_history = []  # 存储状态历史
action_history = []  # 存储动作历史
time_history = []  # 存储时间历史
current_output_array_history = []  # 存储DLL输出历史

# DLL文件路径
DLL_PATH = r'c:\研究生\03设备信息\myRIO\DLL文件 - 参数\Dll1\x64\Debug\Dll1.dll'

# 加载DLL
dll = ctypes.CDLL(DLL_PATH)

# 设置DLL函数的参数类型
# simulate函数参数: 输入数组指针, 当前时间, 时间步长, 输出数组指针
dll.simulate.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_double,
    ctypes.c_double, ctypes.POINTER(ctypes.c_double)
]
dll.simulate.restype = None  # 无返回值

# 初始化输入输出数组
current_input_array = np.zeros((1, action_dim), dtype=np.float64)  # 输入数组
current_output_array = np.zeros(action_dim, dtype=np.float64)  # 输出数组
current_time_sim = 0.0  # 仿真时间初始化

# 打印动作说明
print("\n动作说明：")
print("动作1-3: 三个方向的推力控制 (范围: 0 ~ 0)")
print("动作4-5: 两个方向的姿态控制 (范围: -3 ~ 3)")
print("动作6: 预留控制量 (范围: 0 ~ 0)\n")

# 开始测试
print(f"开始测试，总时长: {TEST_DURATION_SECONDS}秒...")
start_time = time.time()  # 记录开始时间

# 初始状态设置
initial_euler_angles = current_output_array[3:]  # 初始欧拉角
initial_angular_velocities = np.zeros(3)  # 初始角速度
state = np.concatenate([
    current_input_array[0],  # 初始输入
    initial_euler_angles,  # 初始欧拉角
    initial_angular_velocities  # 初始角速度
])

# 主测试循环
for step in range(test_steps):
    # 每100步打印进度
    if step % 100 == 0:
        elapsed_time = time.time() - start_time  # 已用时间
        progress = (step / test_steps) * 100  # 进度百分比
        remaining_time = (elapsed_time / (step + 1)) * (test_steps - step)  # 预计剩余时间
        print(f"进度: {progress:.1f}%, 预计剩余时间: {remaining_time:.1f}秒")

    # 将状态转换为tensor并送入PPO控制器
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(ppo_controller.device)
    
    # 不计算梯度(测试阶段)
    with torch.no_grad():
        action_tensor, _ = ppo_controller.select_action(state_tensor)  # 选择动作
    
    # 将动作转换为numpy数组
    action = action_tensor.cpu().numpy()[0]
    
    # 对动作进行范围限制
    current_input_array[0] = np.clip(action, [0, 0, 0, -15, -15, 0], [0, 0, 0, 15, 15, 0])
    
    # 在15秒时添加干扰到input[3]
    if 14.99 <= current_time_sim <= 15.5:  # 从15秒开始持续干扰
        current_input_array[0, 4] += 6  # 叠加干扰力大小
        current_input_array[0, 4] = np.clip(current_input_array[0, 4], -15, 15)  # 确保在允许范围内
        if 14.99 <= current_time_sim <= 15.05:  # 仅在干扰开始时打印
            print(f"在时间 {current_time_sim:.2f} 秒开始叠加干扰力: 6.0, 当前总控制量: {current_input_array[0, 3]}")

    # 调用DLL进行仿真
    dll.simulate(
        current_input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),  # 输入数组
        current_time_sim,  # 当前时间
        TIME_STEP,  # 时间步长
        current_output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))  # 输出数组
    )
    
    # 保存当前输出
    current_output_array_history.append(current_output_array.copy())
    
    # 计算欧拉角和角速度
    euler_angles = current_output_array[3:]  # 获取欧拉角
    angular_velocities = (euler_angles - state[6:9]) / TIME_STEP if step > 0 else np.zeros(3)  # 计算角速度
    
    # 构建下一状态
    next_state = np.concatenate([
        current_input_array[0],  # 当前输入
        euler_angles,  # 当前欧拉角
        angular_velocities  # 当前角速度
    ])
    
    # 保存历史数据
    state_history.append(state.copy())  # 保存状态
    action_history.append(action.copy())  # 保存动作
    time_history.append(current_time_sim)  # 保存时间
    
    # 更新状态和时间
    state = next_state
    current_time_sim += TIME_STEP

# 测试完成，开始绘图
print("\n测试完成！正在生成图表...")

# 将历史数据转换为numpy数组
state_history = np.array(state_history)
action_history = np.array(action_history)
time_history = np.array(time_history)

# 创建三个子图（总览图）
fig_overview, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

# 绘制位置曲线（从DLL输出获取，前3个值）
ax1.set_title('三轴位置变化')
for i in range(3):
    ax1.plot(time_history, [output[i] for output in current_output_array_history], label=['X轴', 'Y轴', 'Z轴'][i])
ax1.set_xlabel('时间 (秒)')
ax1.set_ylabel('位置 (米)')
ax1.legend()
ax1.grid(True)

# 绘制欧拉角曲线（从DLL输出获取，3-6位置的值）
ax2.set_title('三轴欧拉角变化')
for i in range(3, 6):
    ax2.plot(time_history, [output[i] for output in current_output_array_history], label=['横滚角', '俯仰角', '偏航角'][i-3])
ax2.set_xlabel('时间 (秒)')
ax2.set_ylabel('角度 (度)')
ax2.legend()
ax2.grid(True)

# 绘制动作曲线
ax3.set_title('控制输入')
for i in range(action_dim):
    label = '推力控制' if i < 3 else '姿态控制' if i < 5 else '预留控制'
    ax3.plot(time_history, action_history[:, i], label=f'{label} {i+1}')
ax3.set_xlabel('时间 (秒)')
ax3.set_ylabel('控制量')
ax3.legend()
ax3.grid(True)

# 调整子图间距
plt.tight_layout()

# 保存总览图
fig_overview.savefig('50s_test_output.png', dpi=300, bbox_inches='tight')
plt.close(fig_overview)

# 绘制欧拉角子图
fig_euler, (ax_roll, ax_pitch, ax_yaw) = plt.subplots(3, 1, figsize=(12, 15))

# 横滚角
ax_roll.plot(time_history, [output[3] for output in current_output_array_history], 'r-')
ax_roll.set_title('横滚角变化')
ax_roll.set_xlabel('时间 (秒)')
ax_roll.set_ylabel('角度 (度)')
ax_roll.grid(True)

# 俯仰角
ax_pitch.plot(time_history, [output[4] for output in current_output_array_history], 'g-')
ax_pitch.set_title('俯仰角变化')
ax_pitch.set_xlabel('时间 (秒)')
ax_pitch.set_ylabel('角度 (度)')
ax_pitch.grid(True)

# 偏航角
ax_yaw.plot(time_history, [output[5] for output in current_output_array_history], 'b-')
ax_yaw.set_title('偏航角变化')
ax_yaw.set_xlabel('时间 (秒)')
ax_yaw.set_ylabel('角度 (度)')
ax_yaw.grid(True)

# 调整子图间距
plt.tight_layout()

# 保存欧拉角图表
fig_euler.savefig('euler_angles_detail.png', dpi=300, bbox_inches='tight')
plt.close(fig_euler)

# 绘制控制输入子图
fig_control, ((ax_thrust1, ax_thrust2, ax_thrust3), (ax_attitude1, ax_attitude2, ax_reserve)) = plt.subplots(2, 3, figsize=(18, 12))

# 推力控制
ax_thrust1.plot(time_history, action_history[:, 0], 'r-')
ax_thrust1.set_title('推力控制 1')
ax_thrust1.set_xlabel('时间 (秒)')
ax_thrust1.set_ylabel('控制量')
ax_thrust1.grid(True)

ax_thrust2.plot(time_history, action_history[:, 1], 'g-')
ax_thrust2.set_title('推力控制 2')
ax_thrust2.set_xlabel('时间 (秒)')
ax_thrust2.set_ylabel('控制量')
ax_thrust2.grid(True)

ax_thrust3.plot(time_history, action_history[:, 2], 'b-')
ax_thrust3.set_title('推力控制 3')
ax_thrust3.set_xlabel('时间 (秒)')
ax_thrust3.set_ylabel('控制量')
ax_thrust3.grid(True)

# 姿态控制
ax_attitude1.plot(time_history, action_history[:, 3], 'm-')
ax_attitude1.set_title('姿态控制 1')
ax_attitude1.set_xlabel('时间 (秒)')
ax_attitude1.set_ylabel('控制量')
ax_attitude1.grid(True)

ax_attitude2.plot(time_history, action_history[:, 4], 'c-')
ax_attitude2.set_title('姿态控制 2')
ax_attitude2.set_xlabel('时间 (秒)')
ax_attitude2.set_ylabel('控制量')
ax_attitude2.grid(True)

# 预留控制
ax_reserve.plot(time_history, action_history[:, 5], 'y-')
ax_reserve.set_title('预留控制')
ax_reserve.set_xlabel('时间 (秒)')
ax_reserve.set_ylabel('控制量')
ax_reserve.grid(True)

# 调整子图间距
plt.tight_layout()

# 保存控制输入图表
fig_control.savefig('control_inputs_detail.png', dpi=300, bbox_inches='tight')
plt.close(fig_control)

# 绘制角速度子图
fig_angular_velocity, (ax_roll_rate, ax_pitch_rate, ax_yaw_rate) = plt.subplots(3, 1, figsize=(12, 15))

# 横滚角速度
ax_roll_rate.plot(time_history, state_history[:, 9], 'r-')
ax_roll_rate.set_title('横滚角速度变化')
ax_roll_rate.set_xlabel('时间 (秒)')
ax_roll_rate.set_ylabel('角速度 (度/秒)')
ax_roll_rate.grid(True)

# 俯仰角速度
ax_pitch_rate.plot(time_history, state_history[:, 10], 'g-')
ax_pitch_rate.set_title('俯仰角速度变化')
ax_pitch_rate.set_xlabel('时间 (秒)')
ax_pitch_rate.set_ylabel('角速度 (度/秒)')
ax_pitch_rate.grid(True)

# 偏航角速度
ax_yaw_rate.plot(time_history, state_history[:, 11], 'b-')
ax_yaw_rate.set_title('偏航角速度变化')
ax_yaw_rate.set_xlabel('时间 (秒)')
ax_yaw_rate.set_ylabel('角速度 (度/秒)')
ax_yaw_rate.grid(True)

# 调整子图间距
plt.tight_layout()

# 保存角速度图表
fig_angular_velocity.savefig('angular_velocity_detail.png', dpi=300, bbox_inches='tight')
plt.close(fig_angular_velocity)

# 打印保存的图表信息
print("已保存以下图表：")
print("1. 总览图：'50s_test_output.png'")
print("2. 欧拉角详细图：'euler_angles_detail.png'")
print("3. 控制输入详细图：'control_inputs_detail.png'")
print("4. 角速度详细图：'angular_velocity_detail.png'")