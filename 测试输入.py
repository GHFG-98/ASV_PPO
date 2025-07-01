import numpy as np
import copy

# 定义变量
next_input_array_sim = np.zeros((1, 6), dtype=np.float64)
prev_input_array_sim = np.zeros((1, 6), dtype=np.float64)
time_step = 0.01  # 示例时间步长

# 计算最大变化率
max_rate = 5 * time_step / 0.5

print(f"最大变化率: {max_rate}")

# 测试逐步增加输入值
steps = 100
increase = 0.8  # 测试增量，超过最大变化率

for step in range(steps):
    # 初始化输入值
    next_input_array_sim[0][3] += increase
    
    # 检查更改限制
    for i in [3, 4]:
        delta = next_input_array_sim[0][i] - prev_input_array_sim[0][i]
        if abs(delta) > max_rate:
            next_input_array_sim[0][i] = prev_input_array_sim[0][i] + (max_rate if delta > 0 else -max_rate)

    # 打印结果
    print(f"Step: {step}, Input: {next_input_array_sim[0][3]}")
    
    # 更新
    prev_input_array_sim = next_input_array_sim.copy()
