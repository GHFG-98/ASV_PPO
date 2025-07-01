"""
PPO (Proximal Policy Optimization) 强化学习算法实现 - PyTorch版本

文件功能：
1. 实现了PPO算法的核心组件，包括：
   - RolloutBuffer: 经验回放缓冲区
   - ActorCritic: 演员-评论家网络结构
   - PPOController: PPO算法主控制器
2. 支持的功能：
   - 经验收集与存储
   - 策略评估与优化
   - 模型保存与加载
   - 设备管理(CPU/GPU)

算法特点：
- 使用Clipped Surrogate Objective防止策略更新过大
- 采用Actor-Critic架构
- 支持连续动作空间
- 包含梯度裁剪功能

主要参数：
- gamma: 折扣因子
- eps_clip: PPO clip参数
- K_epochs: 策略更新迭代次数
- lr_actor: 策略网络学习率
- lr_critic: 价值网络学习率
- action_std_init: 初始动作标准差

作者：XXX
创建日期：YYYY-MM-DD
最后修改：YYYY-MM-DD
版本：1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class RolloutBuffer:
    """经验回放缓冲区，用于存储和管理PPO算法的训练数据"""
    def __init__(self):
        # 初始化各种数据列表
        self.actions = []      # 存储动作
        self.states = []       # 存储状态
        self.logprobs = []     # 存储动作的对数概率
        self.rewards = []      # 存储奖励
        self.state_values = [] # 存储状态价值(可能不被主逻辑直接使用)
        self.is_terminals = [] # 存储终止标志
        self.next_states = []  # 存储下一状态(新增的经验结构)
    
    def add(self, state, action, log_prob, reward, next_state, done):
        """向缓冲区添加一条经验数据"""
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.is_terminals.append(done)
        # state_values不直接由此方法填充
        # 在原PPOController.select_action中由policy_old.act填充
        # 如果需要用于GAE或其他优势计算，需要调整

    def extend(self, other_buffer):
        """扩展当前缓冲区，合并另一个缓冲区的数据"""
        self.actions.extend(other_buffer.actions)
        self.states.extend(other_buffer.states)
        self.logprobs.extend(other_buffer.logprobs)
        self.rewards.extend(other_buffer.rewards)
        self.state_values.extend(other_buffer.state_values)
        self.is_terminals.extend(other_buffer.is_terminals)
        self.next_states.extend(other_buffer.next_states) # 新增
    
    def clear(self):
        """清空缓冲区"""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.next_states[:] # 新增

class ActorCritic(nn.Module):
    """演员-评论家网络结构，包含策略网络(actor)和价值网络(critic)"""
    def __init__(self, state_dim, action_dim, action_std_init):
        """初始化网络结构
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            action_std_init: 初始动作标准差
        """
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        # action_var将在set_action_std或__init__中初始化到正确的设备上
        # 目前初始化为占位符，会被覆盖
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
        
        # 演员网络（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),  # 输入层
            nn.Tanh(),                 # 激活函数
            nn.Linear(64, 64),         # 隐藏层
            nn.Tanh(),                 # 激活函数
            nn.Linear(64, action_dim), # 输出层
            nn.Tanh()                  # 使用tanh限制动作范围在[-1, 1]
        )
        
        # 评论家网络（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),  # 输入层
            nn.Tanh(),                 # 激活函数
            nn.Linear(64, 64),         # 隐藏层
            nn.Tanh(),                 # 激活函数
            nn.Linear(64, 1)           # 输出价值
        )
    
    def set_action_std(self, new_action_std):
        """设置新的动作标准差"""
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.actor[0].weight.device)
    
    def forward(self):
        """前向传播(未实现，需要子类实现)"""
        raise NotImplementedError
    
    def act(self, state):
        """根据状态选择动作
        Args:
            state: 输入状态张量
        Returns:
            action: 选择的动作
            action_logprob: 动作的对数概率
        """
        # 检查输入状态是否包含NaN
        if torch.isnan(state).any():
            print("警告: act 方法接收到 NaN 状态，返回零动作和log_prob。")
            device = self.actor[0].weight.device
            return torch.zeros(self.action_dim, device=device), torch.tensor(0.0, device=device)
        
        # 计算动作均值
        action_mean = self.actor(state)
        # 确保action_var与action_mean在同一设备上
        if self.action_var.device != action_mean.device:
            self.action_var = self.action_var.to(action_mean.device)

        # 创建协方差矩阵
        cov_mat = torch.diag(self.action_var) 
        
        # 创建多元正态分布
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        # 采样动作
        action = dist.sample()
        # 计算动作对数概率
        action_logprob = dist.log_prob(action)
        
        return action, action_logprob
    
    def evaluate(self, state, action):
        """评估给定状态和动作的价值
        Args:
            state: 状态张量
            action: 动作张量
        Returns:
            action_logprobs: 动作对数概率
            state_values: 状态价值
            dist_entropy: 分布熵
        """
        # 检查输入是否包含NaN
        if torch.isnan(state).any() or torch.isnan(action).any():
            print("警告: evaluate 方法接收到 NaN 输入，返回默认值。")
            batch_size = state.shape[0]
            device = self.actor[0].weight.device
            return (
                torch.zeros(batch_size, device=device),  # action_logprobs
                torch.zeros(batch_size, 1, device=device),  # state_values
                torch.zeros(batch_size, device=device)   # dist_entropy
            )
        
        # 计算动作均值
        action_mean = self.actor(state)
        
        # 确保action_var在正确设备上
        action_var_eval = self.action_var.to(action_mean.device).expand_as(action_mean)
        # 创建协方差矩阵(批处理版本)
        cov_mat = torch.diag_embed(action_var_eval) 
        
        # 创建多元正态分布
        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        
        # 计算动作对数概率
        action_logprobs = dist.log_prob(action)
        # 计算分布熵
        dist_entropy = dist.entropy()
        # 计算状态价值
        state_values = self.critic(state) 
        
        return action_logprobs, state_values, dist_entropy

class PPOController:
    """PPO算法主控制器，管理整个PPO训练过程"""
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6, max_grad_norm=0.5, device=torch.device('cpu')):
        """初始化PPO控制器
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr_actor: 策略网络学习率
            lr_critic: 价值网络学习率
            gamma: 折扣因子
            K_epochs: 策略更新迭代次数
            eps_clip: PPO clip参数
            action_std_init: 初始动作标准差
            max_grad_norm: 最大梯度范数(用于梯度裁剪)
            device: 计算设备(CPU/GPU)
        """
        self.gamma = gamma         # 折扣因子
        self.eps_clip = eps_clip   # PPO clip参数
        self.K_epochs = K_epochs   # 策略更新迭代次数
        self.max_grad_norm = max_grad_norm  # 最大梯度范数
        self.device = device       # 计算设备
        
        # 初始化经验回放缓冲区
        self.buffer = RolloutBuffer()
        
        # 初始化策略网络
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
        # 设置优化器(分别为actor和critic设置不同学习率)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        # 初始化旧策略网络(用于计算旧策略的概率)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval() # 设置旧策略为评估模式
        
        # 均方误差损失函数
        self.MseLoss = nn.MSELoss()
    
    def set_action_std(self, new_action_std):
        """设置新的动作标准差"""
        self.policy.set_action_std(new_action_std) # 这会确保action_var在正确设备上
        self.policy_old.set_action_std(new_action_std)
    
    def select_action(self, state_tensor):
        """根据状态选择动作
        Args:
            state_tensor: 状态张量(已在正确设备上)
        Returns:
            action_tensor: 选择的动作
            action_log_prob_tensor: 动作的对数概率
        """
        if torch.isnan(state_tensor).any():
            print("警告: PPOController.select_action 状态包含 NaN 值，返回零动作和log_prob。")
            return torch.zeros(self.policy.action_dim, device=self.device), torch.tensor(0.0, device=self.device)
        
        with torch.no_grad():  # 不计算梯度
            # 使用旧策略网络选择动作
            action_tensor, action_log_prob_tensor = self.policy_old.act(state_tensor)
            # 主脚本会处理将数据添加到缓冲区:
            # ppo_controller.buffer.add(state, action, log_prob, reward, next_state, done)
            # 所以此方法只需返回动作和其对数概率
        
        return action_tensor, action_log_prob_tensor
    
    def update(self):
        """更新策略网络和价值网络"""
        # 计算蒙特卡洛估计的回报
        rewards_list = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # 计算折扣回报
            discounted_reward = float(reward) + (self.gamma * discounted_reward)
            rewards_list.insert(0, discounted_reward)
        
        # 归一化回报
        rewards_tensor = torch.tensor(rewards_list, dtype=torch.float32).to(self.device)
        if rewards_tensor.numel() > 1:  # 多个奖励
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7)
        elif rewards_tensor.numel() == 1:  # 单个奖励
            rewards_tensor = torch.zeros_like(rewards_tensor)
        else: # 无奖励
            print("警告: PPO update 时 rewards 为空，跳过更新。")
            self.buffer.clear()
            return

        # 过滤掉状态包含NaN的经验
        valid_indices = [i for i, s_np in enumerate(self.buffer.states) if not np.isnan(s_np).any()]
        if not valid_indices:
            print("警告: PPO update 时所有状态都包含 NaN，跳过更新。")
            self.buffer.clear()
            return

        # 根据有效索引获取数据
        old_states_np_list = [self.buffer.states[i] for i in valid_indices]
        old_actions_np_list = [self.buffer.actions[i] for i in valid_indices]
        old_logprobs_np_list = [self.buffer.logprobs[i] for i in valid_indices]
        
        # 过滤奖励张量
        rewards_tensor = rewards_tensor[torch.tensor(valid_indices, dtype=torch.long, device=self.device)]

        if not old_states_np_list: # 过滤后无有效数据
            print("警告: PPO update 过滤后没有有效数据，跳过更新。")
            self.buffer.clear()
            return

        # 将numpy数组转换为张量
        old_states_tensor = torch.tensor(np.array(old_states_np_list), dtype=torch.float32).to(self.device)
        old_actions_tensor = torch.tensor(np.array(old_actions_np_list), dtype=torch.float32).to(self.device)
        old_logprobs_tensor = torch.tensor(np.array(old_logprobs_np_list), dtype=torch.float32).to(self.device)

        # 重新归一化过滤后的奖励
        if rewards_tensor.numel() > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-7)
        elif rewards_tensor.numel() == 1:
            rewards_tensor = torch.zeros_like(rewards_tensor)
        else: # 过滤后无奖励
            print("警告: PPO update 时过滤后的 rewards 为空，跳过更新。")
            self.buffer.clear()
            return
        
        # 优化策略K个epoch
        for _ in range(self.K_epochs):
            # 评估旧动作和状态价值
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_tensor, old_actions_tensor)
            state_values = torch.squeeze(state_values) # 调整形状以匹配奖励张量
            
            # 计算优势函数
            advantages = rewards_tensor - state_values.detach()
            
            # 计算比率(新策略概率/旧策略概率)
            ratios = torch.exp(logprobs - old_logprobs_tensor.detach())

            # 计算替代损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # 计算最终损失(包含clip目标、价值函数误差和熵正则项)
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards_tensor) - 0.01*dist_entropy
            
            # 梯度下降步骤
            self.optimizer.zero_grad()
            loss.mean().backward()
            if self.max_grad_norm is not None:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # 将新策略参数复制到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.eval() # 确保旧策略保持评估模式
        
        # 清空缓冲区
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        """保存模型参数到文件"""
        torch.save(self.policy_old.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path):
        """从文件加载模型参数"""
        # 加载模型权重，确保在正确设备上
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy_old.eval()
        self.policy.eval() # 加载后设置为评估模式，训练时再切换为训练模式

    def to(self, device):
        """将PPO控制器的策略和相关张量移动到指定设备"""
        self.device = device
        self.policy.to(device)
        self.policy_old.to(device)
        # 重新设置action_var以确保设备一致性
        self.policy.set_action_std(self.policy.action_var.sqrt()[0].item())
        self.policy_old.set_action_std(self.policy_old.action_var.sqrt()[0].item())
        return self
