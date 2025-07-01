import ctypes

import numpy as np  # 保留numpy用于ctypes转换

import matplotlib.pyplot as plt

# DLL路径
DLL_PATH = r'c:\研究生\03设备信息\myRIO\DLL文件 - 参数\Dll3\x64\Debug\Dll1.dll'

# 加载DLL
try:
    dll = ctypes.CDLL(DLL_PATH)
    dll.simulate.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # 输入数组
        ctypes.c_double,                  # 当前时间
        ctypes.c_double,                  # 时间步长
        ctypes.POINTER(ctypes.c_double)   # 输出数组
    ]
    dll.simulate.restype = None
    print("DLL加载成功")
except Exception as e:
    print(f"加载DLL失败: {e}")
    exit(1)

# 初始化参数
input_size = 6   # 输入维度
output_size = 6  # 输出维度
time_step = 0.01 # 时间步长

# 创建输入和输出数组
input_array = np.zeros((1, input_size), dtype=np.float64)
output_array = np.zeros(output_size, dtype=np.float64)
output_history = []
time_history = []

# 设置一些简单的输入值进行测试
input_array[0, 0] = 0.0  # 设置第一个输入为1.0

# 执行仿真
current_time = 0.0
end_time = 20.0  # 运行10秒

print("\n初始状态:")
print(f"输入数组: {input_array.flatten()}")
print(f"输出数组: {output_array}")

while current_time < end_time:
    # 在15秒时添加6值瞬时干扰
    if 14.99 <= current_time <= 20:
        input_array[0, 4] = 6.0  # 在input[3]添加6值干扰
    else:
        input_array[0, 4] = 0.0  # 恢复为0
    # 检查欧拉角是否超出裕度范围
    euler_angles = output_array[:3]
    euler_margin_low = -10.0  # 欧拉角裕度下限
    euler_margin_high = 10.0  # 欧拉角裕度上限
    if any(angle < euler_margin_low or angle > euler_margin_high for angle in euler_angles):
        # 计算控制输入（简单比例控制示例）
        error = np.clip(euler_angles, euler_margin_low, euler_margin_high) - euler_angles
        input_array[0, 4] = np.sum(np.abs(error)) * 0.1  # 比例增益设为0.1
    else:
        input_array[0, 4] = 0.0  # 裕度范围内输入为0
    
    # 调用DLL进行仿真
    dll.simulate(
        input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        current_time,
        time_step,
        output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    # 每100步打印一次状态（减少IO开销）
    if len(time_history) % 100 == 0:
        print(f"\n时间 {current_time:.2f} 秒:")
        print(f"输入数组: {input_array.flatten()}")
        print(f"输出数组: {output_array}")
    
    # 记录输出数据
    output_history.append(output_array.copy())
    time_history.append(current_time)
    
    # 检查输出是否包含NaN
    if np.isnan(output_array).any():
        print("警告: 输出数组包含NaN值")
        break
    
    current_time += time_step

# 检查输出是否包含NaN
if np.isnan(output_array).any():
    print("\n警告: 输出数组包含NaN值")
else:
    print("\n输出数组正常，没有NaN值")
    
# 保存输出数据图表
output_history = np.array(output_history)
plt.figure(figsize=(12, 8))
for i in range(output_size):
    plt.plot(time_history, output_history[:, i], label=f'输出{i+1}')
plt.xlabel('时间 (秒)')
plt.ylabel('输出值')
plt.title('DLL输出随时间变化')
plt.legend()
plt.grid(True)
plt.savefig('dll_output_disturb_plot2.png')
print("已保存输出图表: dll_output_disturb_plot.png")