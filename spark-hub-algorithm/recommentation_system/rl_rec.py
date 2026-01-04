import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DQNRecommender(nn.Module):
    """
    基于深度Q网络(DQN)的强化学习推荐算法
    
    使用深度强化学习来优化推荐策略，最大化长期用户满意度
    实现了Deep Q-Network (DQN) 的核心思想，包括经验回放和目标网络
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, gamma=0.99, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                 learning_rate=0.001, memory_size=10000, batch_size=64):
        """
        初始化DQN推荐模型
        
        参数:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度（物品数量）
            hidden_dim (int): 隐藏层维度
            gamma (float): 折扣因子
            epsilon (float): 探索率初始值
            epsilon_min (float): 探索率最小值
            epsilon_decay (float): 探索率衰减因子
            learning_rate (float): 学习率
            memory_size (int): 经验回放缓冲区大小
            batch_size (int): 批量大小
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=memory_size)
        
        # 主网络
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 目标网络
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # 初始化目标网络
        self.update_target_network()
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def update_target_network(self):
        """更新目标网络参数"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def get_action(self, state, available_items=None):
        """
        根据当前状态选择动作（推荐物品）
        
        参数:
            state (torch.Tensor): 当前状态
            available_items (list): 可选物品列表，默认为None表示所有物品可选
            
        返回:
            int: 推荐的物品ID
        """
        # epsilon-贪婪策略
        if random.random() < self.epsilon:
            if available_items is not None:
                return random.choice(available_items)
            else:
                return random.randint(0, self.action_dim - 1)
        
        # 使用Q网络预测最佳动作
        with torch.no_grad():
            q_values = self.q_network(state)
            
            # 如果有限制可选物品，则将不可选物品的Q值设为负无穷
            if available_items is not None:
                mask = torch.ones_like(q_values) * float('-inf')
                mask[available_items] = 0
                q_values = q_values + mask
            
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """
        将经验存储到回放缓冲区
        
        参数:
            state (torch.Tensor): 当前状态
            action (int): 执行的动作
            reward (float): 获得的奖励
            next_state (torch.Tensor): 下一个状态
            done (bool): 是否结束
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """
        从经验回放缓冲区中采样并训练模型
        
        返回:
            float: 训练损失
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # 随机采样
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为张量
        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def train_model(self, env, num_episodes=100, target_update_freq=10, device='cuda'):
        """
        训练模型
        
        参数:
            env: 环境对象，需要实现reset(), step(action)方法
            num_episodes (int): 训练回合数
            target_update_freq (int): 目标网络更新频率
            device (str): 训练设备
            
        返回:
            list: 训练奖励历史
        """
        self.to(device)
        rewards_history = []
        
        for episode in range(num_episodes):
            state = env.reset()
            state = torch.FloatTensor(state).to(device)
            total_reward = 0
            done = False
            
            while not done:
                # 选择动作
                action = self.get_action(state)
                
                # 执行动作
                next_state, reward, done, _ = env.step(action)
                next_state = torch.FloatTensor(next_state).to(device)
                
                # 存储经验
                self.remember(state, action, reward, next_state, done)
                
                # 训练模型
                loss = self.replay()
                
                # 更新状态
                state = next_state
                total_reward += reward
            
            # 更新目标网络
            if episode % target_update_freq == 0:
                self.update_target_network()
            
            rewards_history.append(total_reward)
            print(f"Episode {episode+1}/{num_episodes}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")
        
        return rewards_history
    
    def predict(self, state, k=5, available_items=None):
        """
        预测用户可能感兴趣的物品
        
        参数:
            state (torch.Tensor): 用户状态
            k (int): 推荐物品数量
            available_items (list): 可选物品列表
            
        返回:
            list: 推荐的物品ID列表
        """
        self.eval()
        with torch.no_grad():
            q_values = self.q_network(state)
            
            # 如果有限制可选物品，则将不可选物品的Q值设为负无穷
            if available_items is not None:
                mask = torch.ones_like(q_values) * float('-inf')
                mask[available_items] = 0
                q_values = q_values + mask
            
            # 获取top-k物品
            _, indices = torch.topk(q_values, k)
            return indices.cpu().numpy().tolist()


class RecEnv:
    """
    推荐系统环境
    
    模拟用户与推荐系统的交互过程
    """
    def __init__(self, user_features, item_features, ratings_matrix, max_steps=20):
        """
        初始化推荐环境
        
        参数:
            user_features (numpy.ndarray): 用户特征矩阵
            item_features (numpy.ndarray): 物品特征矩阵
            ratings_matrix (numpy.ndarray): 用户-物品评分矩阵
            max_steps (int): 最大交互步数
        """
        self.user_features = user_features
        self.item_features = item_features
        self.ratings_matrix = ratings_matrix
        self.max_steps = max_steps
        
        self.num_users = user_features.shape[0]
        self.num_items = item_features.shape[0]
        
        self.current_user = None
        self.current_step = 0
        self.interacted_items = set()
    
    def reset(self):
        """
        重置环境
        
        返回:
            numpy.ndarray: 初始状态
        """
        # 随机选择一个用户
        self.current_user = np.random.randint(0, self.num_users)
        
        # 重置交互历史
        self.interacted_items = set()
        self.current_step = 0
        
        # 构建初始状态
        state = self._build_state()
        
        return state
    
    def step(self, action):
        """
        执行推荐动作
        
        参数:
            action (int): 推荐的物品ID
            
        返回:
            tuple: (next_state, reward, done, info)
        """
        # 检查动作是否有效
        if action < 0 or action >= self.num_items:
            return self._build_state(), -1, True, {"error": "Invalid action"}
        
        # 检查是否重复推荐
        if action in self.interacted_items:
            return self._build_state(), -0.5, False, {"info": "Item already recommended"}
        
        # 记录交互
        self.interacted_items.add(action)
        self.current_step += 1
        
        # 计算奖励（使用真实评分）
        reward = self.ratings_matrix[self.current_user, action]
        
        # 如果评分为0，表示用户未交互过该物品，给予较小的负奖励
        if reward == 0:
            reward = -0.1
        
        # 检查是否结束
        done = self.current_step >= self.max_steps
        
        # 构建下一个状态
        next_state = self._build_state()
        
        return next_state, reward, done, {}
    
    def _build_state(self):
        """
        构建当前状态表示
        
        返回:
            numpy.ndarray: 状态向量
        """
        # 状态包含：用户特征 + 最近交互的物品特征 + 交互历史
        user_feat = self.user_features[self.current_user]
        
        # 如果没有交互历史，使用零向量
        if len(self.interacted_items) == 0:
            recent_items_feat = np.zeros_like(self.item_features[0])
        else:
            # 使用最近交互的物品特征的平均值
            recent_items = list(self.interacted_items)[-5:]  # 最多取最近5个
            recent_items_feat = np.mean([self.item_features[item] for item in recent_items], axis=0)
        
        # 构建交互历史向量（one-hot编码）
        history = np.zeros(self.num_items)
        for item in self.interacted_items:
            history[item] = 1
        
        # 组合状态向量
        state = np.concatenate([user_feat, recent_items_feat, history])
        
        return state 