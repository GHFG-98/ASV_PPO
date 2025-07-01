import ctypes
import numpy as np
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

# 基于第一个程序的输入配置
input_array[0, 3] = 1.0  # 设置第一个输入为1.0

# 执行仿真
current_time = 0.0
end_time = 50.0  # 运行10秒

print("\n初始状态:")
print(f"输入数组: {input_array.flatten()}")
print(f"输出数组: {output_array}")

while current_time < end_time:
    # 调用DLL进行仿真
    dll.simulate(
        input_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        current_time,
        time_step,
        output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    )
    
    # 每个时间步都打印状态
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
plt.savefig('dll_output_plot.png')
print("已保存输出图表: dll_output_plot.png")
