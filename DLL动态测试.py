import ctypes
import numpy as np
import time
import matplotlib.pyplot as plt

# --- 配置参数 ---
DLL_PATH = r'c:\研究生\03设备信息\myRIO\DLL文件 - 参数\Dll3\x64\Debug\Dll1.dll'  # 替换为你的实际路径
TIME_STEP = 0.01  # 时间步长
TEST_STEPS = 200  # 测试运行步数
INPUT_SIZE = 6    # 输入维度
OUTPUT_SIZE = 6   # 输出维度

# --- 加载 DLL ---
try:
    dll = ctypes.CDLL(DLL_PATH)
except Exception as e:
    print(f"❌ 加载 DLL 失败: {e}")
    exit(1)

# 设置函数原型：void simulate(double* input, double t, double dt, double* output)
dll.simulate.argtypes = [
    ctypes.POINTER(ctypes.c_double),  # input array
    ctypes.c_double,                  # current time
    ctypes.c_double,                  # time step
    ctypes.POINTER(ctypes.c_double)   # output array
]
dll.simulate.restype = None

print("✅ DLL 加载成功！开始测试...")

# --- 初始化输入输出数组 ---
input_array = np.zeros(INPUT_SIZE, dtype=np.float64)
output_array = np.zeros(OUTPUT_SIZE, dtype=np.float64)

# 可以设置初始输入（例如控制信号）
input_array[3] = 0.0  # 假设第四个输入是某个控制变量

current_time = 0.0

# --- 存储结果以便分析 ---
history = []

# --- 开始测试循环 ---
start_time = time.time()
for step in range(TEST_STEPS):
    # 调用 DLL 的 simulate 函数
    dll.simulate(
        input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(current_time),
        ctypes.c_double(TIME_STEP),
        output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )

    # 记录当前状态
    euler_angles = output_array[3:]  # 假设后三个是欧拉角
    angular_velocities = (output_array[3:] - input_array[3:]) / TIME_STEP if step > 0 else np.zeros(3)

    print(f"[Step {step}] Output: {output_array}")
    print(f"  Euler Angles: {euler_angles}, Angular Velocities: {angular_velocities}")

    history.append({
        'time': current_time,
        'input': input_array.copy(),
        'output': output_array.copy(),
        'euler': euler_angles.copy(),
        'angular_velocity': angular_velocities.copy()
    })

    # 更新输入为当前输出的一部分（可以自定义逻辑）
    input_array[:3] = 1  # 前三个输入设为0
    input_array[3:] = 0  # 后三个也更新

    current_time += TIME_STEP

end_time = time.time()

print("\n✅ 测试完成！总耗时:", end_time - start_time, "秒")