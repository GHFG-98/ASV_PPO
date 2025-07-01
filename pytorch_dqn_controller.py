import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# 定义神经网络模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义DQN控制器
class DQNController:
    def __init__(self, state_size=12, action_size=6, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # 检测并设置设备 (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQNController] Using device: {self.device}")

        self.model = DQN(state_size, action_size).to(self.device) # 将模型移动到设备
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # 探索：返回随机动作 (需要确保动作在合理范围内，这里假设是[-1, 1])
            # return np.random.rand(1, self.action_size) * 2 - 1 
            # 或者返回零向量作为探索？这取决于具体问题
            return np.zeros((1, self.action_size)) # 暂时返回零向量作为探索
        
        # 利用：使用模型预测动作
        state = torch.FloatTensor(state).to(self.device) # 将状态移动到设备
        with torch.no_grad():
            act_values = self.model(state)
        # 对于连续动作空间，直接返回网络输出作为动作
        # 需要确保输出在DLL期望的范围内，可能需要裁剪或缩放
        # return act_values.cpu().numpy() # 将结果移回CPU并转为numpy
        # 假设DLL期望的输入范围是[-1, 1]，使用tanh激活或手动裁剪
        action = act_values.cpu().numpy()
        # return np.clip(action, -1.0, 1.0) # 示例：裁剪到[-1, 1]
        return action # 暂时不裁剪，假设DLL能处理任意值或后续步骤处理

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        processed_count = 0
        for state, action, reward, next_state, done in minibatch:
            processed_count += 1
            # --- 更详细的 NaN 检查 --- 
            nan_detected_input = False
            try:
                # 检查 state, action, reward, next_state 是否包含 NaN (假设它们是 numpy arrays 或 scalars)
                if isinstance(state, np.ndarray) and np.isnan(state).any(): print(f"!!! [Replay Batch {processed_count}/{batch_size}] NaN in state: {state} !!!"); nan_detected_input = True
                if isinstance(action, np.ndarray) and np.isnan(action).any(): print(f"!!! [Replay Batch {processed_count}/{batch_size}] NaN in action: {action} !!!"); nan_detected_input = True
                if isinstance(reward, (int, float)) and np.isnan(reward): print(f"!!! [Replay Batch {processed_count}/{batch_size}] NaN in reward: {reward} !!!"); nan_detected_input = True
                if isinstance(next_state, np.ndarray) and np.isnan(next_state).any(): print(f"!!! [Replay Batch {processed_count}/{batch_size}] NaN in next_state: {next_state} !!!"); nan_detected_input = True
            except Exception as e:
                print(f"Error during input NaN check: {e}")
                nan_detected_input = True # Treat error as potential NaN source

            if nan_detected_input:
                print(f"NaN detected in input data for batch item {processed_count}. Skipping this item.")
                continue # 跳过这个有问题的样本

            # 将所有数据移动到设备
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device) 
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device) 
            action = torch.FloatTensor(action).to(self.device) # 动作已经是(1, action_size)或(action_size,)?
            reward = torch.FloatTensor([reward]).to(self.device) 
            done = torch.FloatTensor([done]).to(self.device) 

            # Predict Q-values (or state-values in this adapted context) for current and next states
            try:
                current_q_values = self.model(state)
                if torch.isnan(current_q_values).any():
                    print(f"!!! [Replay Batch {processed_count}/{batch_size}] NaN detected in model prediction (current_q_values): {current_q_values} !!!")
                    print(f"State input was: {state}")
                    continue # 跳过
            except Exception as e:
                print(f"Error predicting current_q_values: {e}")
                continue

            try:
                with torch.no_grad(): # Target network prediction should not have gradients
                    next_q_values = self.model(next_state)
                    if torch.isnan(next_q_values).any():
                        print(f"!!! [Replay Batch {processed_count}/{batch_size}] NaN detected in model prediction for next_state: {next_q_values} !!!")
                        print(f"Next state input was: {next_state}")
                        continue # 跳过
            except Exception as e:
                print(f"Error predicting next_q_values: {e}")
                continue

            # Calculate target value
            try:
                target_next_value = torch.max(next_q_values) # Max returns a single value
                if torch.isnan(target_next_value):
                    print(f"!!! [Replay Batch {processed_count}/{batch_size}] NaN detected in max(next_q_values): {target_next_value} !!!")
                    continue # 跳过
                
                target = reward + (1 - done) * self.gamma * target_next_value
                if torch.isnan(target).any(): # Target might be a tensor if reward/done were tensors
                    print(f"!!! [Replay Batch {processed_count}/{batch_size}] NaN detected in target calculation: {target} !!!")
                    print(f"Reward: {reward}, Gamma: {self.gamma}, MaxNextQ: {target_next_value}, Done: {done}")
                    continue # 跳过
            except Exception as e:
                print(f"Error calculating target: {e}")
                continue 

            self.optimizer.zero_grad()
            try:
                # Calculate loss between the predicted values for the current state 
                # and the calculated target value.
                # WARNING: This loss calculation is standard for discrete DQN but questionable for continuous actions.
                # It compares the entire action vector prediction with a single scalar target expanded.
                loss = self.criterion(current_q_values, target.expand_as(current_q_values))
                
                if torch.isnan(loss):
                    print(f"!!! [Replay Batch {processed_count}/{batch_size}] NaN detected in loss calculation: {loss} !!!")
                    print(f"Current Q Values: {current_q_values}\nTarget Expanded: {target.expand_as(current_q_values)}")
                    continue # 跳过

                loss.backward()
                # --- 添加梯度裁剪 ---
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # 裁剪梯度范数到 1.0
                self.optimizer.step()
            except Exception as e:
                 print(f"Error during backpropagation: {e}")
                 print(f"state shape: {state.shape}, type: {state.dtype}")
                 print(f"current_q_values shape: {current_q_values.shape}, type: {current_q_values.dtype}")
                 print(f"target shape: {target.shape}, type: {target.dtype}")
                 continue # Skip if error
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # 可能需要添加保存和加载模型的方法
    # def save_model(self, path):
    #     torch.save(self.model.state_dict(), path)

    # def load_model(self, path):
    #     self.model.load_state_dict(torch.load(path, map_location=self.device))
    #     self.model.eval() # 设置为评估模式