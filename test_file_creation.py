import os

# 测试文件创建功能
PLOT_DIR = "episode_plots_ppo_mp"
noise_file = os.path.join(PLOT_DIR, "noise_data_all.csv")

try:
    # 确保目录存在
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # 测试写入文件
    with open(noise_file, 'a') as f:
        f.write("1,0.5,0.5\n")
    
    print(f"成功写入文件: {noise_file}")
    print(f"文件存在: {os.path.exists(noise_file)}")
except Exception as e:
    print(f"发生错误: {str(e)}")