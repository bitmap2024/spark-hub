import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE) 推荐算法
    
    使用多任务学习和专家混合系统进行推荐
    论文：Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts
    """
    def __init__(self, input_dim, num_experts=4, expert_dim=64, num_tasks=2, gate_dim=32):
        """
        初始化MMoE模型
        
        参数:
            input_dim (int): 输入特征维度
            num_experts (int): 专家数量
            expert_dim (int): 专家网络隐藏层维度
            num_tasks (int): 任务数量
            gate_dim (int): 门控网络隐藏层维度
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.num_tasks = num_tasks
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, expert_dim),
                nn.ReLU()
            ) for _ in range(num_experts)
        ])
        
        # 任务门控网络
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, gate_dim),
                nn.ReLU(),
                nn.Linear(gate_dim, num_experts),
                nn.Softmax(dim=1)
            ) for _ in range(num_tasks)
        ])
        
        # 任务特定层
        self.task_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, expert_dim // 2),
                nn.ReLU(),
                nn.Linear(expert_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征，形状为 (batch_size, input_dim)
            
        返回:
            list: 每个任务的预测结果
        """
        # 获取专家输出
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, expert_dim)
        
        # 计算每个任务的门控权重并获取最终输出
        final_outputs = []
        for task_id in range(self.num_tasks):
            gate_output = self.gates[task_id](x)  # (batch_size, num_experts)
            gate_output = gate_output.unsqueeze(-1)  # (batch_size, num_experts, 1)
            
            # 加权组合专家输出
            task_expert_output = torch.sum(expert_outputs * gate_output, dim=1)  # (batch_size, expert_dim)
            
            # 任务特定层
            task_output = self.task_layers[task_id](task_expert_output)
            final_outputs.append(task_output)
            
        return final_outputs
    
    def calculate_loss(self, predictions, targets, task_weights=None):
        """
        计算多任务损失
        
        参数:
            predictions (list): 每个任务的预测结果
            targets (list): 每个任务的目标值
            task_weights (list): 任务权重，默认为None表示等权重
            
        返回:
            torch.Tensor: 加权总损失
        """
        if task_weights is None:
            task_weights = [1.0] * self.num_tasks
            
        total_loss = 0
        task_losses = []
        
        for task_id in range(self.num_tasks):
            task_loss = F.binary_cross_entropy(
                predictions[task_id],
                targets[task_id],
                reduction='mean'
            )
            weighted_loss = task_loss * task_weights[task_id]
            total_loss += weighted_loss
            task_losses.append(task_loss.item())
            
        return total_loss, task_losses
    
    def predict(self, x):
        """
        预测
        
        参数:
            x (torch.Tensor): 输入特征
            
        返回:
            list: 每个任务的预测结果
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
            return [p.cpu().numpy() for p in predictions]
    
    def train_model(self, train_loader, optimizer, num_epochs=10, task_weights=None, device='cuda'):
        """
        训练模型
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            optimizer: 优化器
            num_epochs (int): 训练轮数
            task_weights (list): 任务权重
            device (str): 训练设备
            
        返回:
            list: 训练损失历史
        """
        self.to(device)
        history = {f'task_{i}_loss': [] for i in range(self.num_tasks)}
        history['total_loss'] = []
        
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            task_total_losses = [0] * self.num_tasks
            
            for batch in train_loader:
                features = batch['features'].to(device)
                targets = [batch[f'task_{i}_target'].to(device) for i in range(self.num_tasks)]
                
                optimizer.zero_grad()
                predictions = self.forward(features)
                
                loss, task_losses = self.calculate_loss(predictions, targets, task_weights)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                for i, task_loss in enumerate(task_losses):
                    task_total_losses[i] += task_loss
            
            # 记录损失
            avg_total_loss = total_loss / len(train_loader)
            history['total_loss'].append(avg_total_loss)
            
            for i in range(self.num_tasks):
                avg_task_loss = task_total_losses[i] / len(train_loader)
                history[f'task_{i}_loss'].append(avg_task_loss)
            
            # 打印训练进度
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Total Loss: {avg_total_loss:.4f}")
            for i in range(self.num_tasks):
                print(f"Task {i} Loss: {history[f'task_{i}_loss'][-1]:.4f}")
            print()
        
        return history 