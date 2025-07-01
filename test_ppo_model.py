import torch
import numpy as np
from pytorch_ppo_controller import PPOController
import matplotlib.pyplot as plt
import ctypes
import time

# 加载训练好的模型
MODEL_PATH = r'C:\研究生\03设备信息\myRIO\DLL文件 - 参数\episode_plots_ppo_mp\ppo_mp_model.pth'  # 修改为你的模型路径

# 初始化PPO控制器 (参数需要与训练时一致)
state_dim = 12  # 输入6 + 欧拉角3 + 角速度3
action_dim = 6  # 动作维度
TEST_DURATION_SECONDS = 50  # 测试持续时间
TIME_STEP = 0.01  # 时间步长

# 在import后添加字体设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ppo_controller = PPOController(
    state_dim=state_dim,
    action_dim=action_dim,
    lr_actor=0.0003,
    lr_critic=0.001,
    gamma=0.99,
    K_epochs=10,
    eps_clip=0.2,
    action_std_init=0.1,
    max_grad_norm=0.5,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 加载模型权重
ppo_controller.load(MODEL_PATH)
print(f"成功加载模型: {MODEL_PATH}")

# 测试模型
test_steps = int(TEST_DURATION_SECONDS / TIME_STEP)
state_history = []
action_history = []
time_history = []
current_output_array_history = []  # 添加DLL输出历史记录列表

# 在PPO控制器初始化之前添加DLL配置
DLL_PATH = r'c:\研究生\03设备信息\myRIO\DLL文件 - 参数\Dll1\x64\Debug\Dll1.dll'

# 加载DLL
dll = ctypes.CDLL(DLL_PATH)
dll.simulate.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_double,
    ctypes.c_double, ctypes.POINTER(ctypes.c_double)
]
dll.simulate.restype = None

# 在测试循环中替换随机状态更新为DLL交互
current_input_array = np.zeros((1, action_dim), dtype=np.float64)
current_output_array = np.zeros(action_dim, dtype=np.float64)
current_time_sim = 0.0

print("\n动作说明：")
print("动作1-3: 三个方向的推力控制 (范围: 0 ~ 0)")
print("动作4-5: 两个方向的姿态控制 (范围: -3 ~ 3)")
print("动作6: 预留控制量 (范围: 0 ~ 0)\n")

print(f"开始测试，总时长: {TEST_DURATION_SECONDS}秒...")
start_time = time.time()

# 在测试循环前添加初始状态初始化
# 执行一次DLL调用获取初始状态
dll.simulate(
    current_input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    current_time_sim,
    TIME_STEP,
    current_output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
)

# 构建初始状态
initial_euler_angles = current_output_array[3:]
initial_angular_velocities = np.zeros(3)
state = np.concatenate([
    current_input_array[0],  # 初始动作（全零）
    initial_euler_angles,    # 初始欧拉角
    initial_angular_velocities  # 初始角速度
])

print("初始状态：")
print(f"欧拉角: {initial_euler_angles}")
print(f"角速度: {initial_angular_velocities}\n")

for step in range(test_steps):
    if step % 100 == 0:  # 每100步显示一次进度
        elapsed_time = time.time() - start_time
        progress = (step / test_steps) * 100
        remaining_time = (elapsed_time / (step + 1)) * (test_steps - step)
        print(f"进度: {progress:.1f}%, 预计剩余时间: {remaining_time:.1f}秒")

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(ppo_controller.device)
    
    with torch.no_grad():
        action_tensor, _ = ppo_controller.select_action(state_tensor)
    
    action = action_tensor.cpu().numpy()[0]
    
    # 应用动作限制
    current_input_array[0] = np.clip(action, [0, 0, 0, -3, -3, 0], [0, 0, 0, 3, 3, 0])
    
    # 执行DLL仿真
    dll.simulate(
        current_input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        current_time_sim,
        TIME_STEP,
        current_output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    # 记录DLL输出数据
    current_output_array_history.append(current_output_array.copy())
    
    # 构建下一个状态（使用DLL输出）
    euler_angles = current_output_array[3:]
    angular_velocities = (euler_angles - state[6:9]) / TIME_STEP if step > 0 else np.zeros(3)
    next_state = np.concatenate([
        current_input_array[0],  # 当前动作
        euler_angles,            # 欧拉角
        angular_velocities       # 角速度
    ])
    
    # 记录数据
    state_history.append(state.copy())
    action_history.append(action.copy())
    time_history.append(current_time_sim)
    
    # 更新状态和时间
    state = next_state
    current_time_sim += TIME_STEP

print("\n测试完成！正在生成图表...")

# 转换为numpy数组以便绘图
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

print("已保存以下图表：")
print("1. 总览图：'50s_test_output.png'")
print("2. 欧拉角详细图：'euler_angles_detail.png'")
print("3. 控制输入详细图：'control_inputs_detail.png'")
print("4. 角速度详细图：'angular_velocity_detail.png'")