import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Any, Optional

class PLE(nn.Module):
    """
    Progressive Layered Extraction (PLE) 多目标推荐模型
    
    论文: Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
    
    PLE模型通过设计专家共享层和任务特定层，平衡知识共享和任务专有特征提取，
    比MMoE有更好的任务隔离性和知识共享能力
    """
    def __init__(self, 
                 input_dim: int, 
                 shared_experts: int = 4, 
                 task_specific_experts: int = 3, 
                 expert_dim: int = 64, 
                 num_tasks: int = 3, 
                 num_layers: int = 2,
                 gate_dim: int = 32):
        """
        初始化PLE模型
        
        参数:
            input_dim (int): 输入特征维度
            shared_experts (int): 共享专家数量
            task_specific_experts (int): 每个任务特定专家数量
            expert_dim (int): 专家网络隐藏层维度
            num_tasks (int): 任务数量
            num_layers (int): CGC层数量
            gate_dim (int): 门控网络隐藏层维度
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.shared_experts = shared_experts
        self.task_specific_experts = task_specific_experts
        self.expert_dim = expert_dim
        self.num_tasks = num_tasks
        self.num_layers = num_layers
        
        # 创建多层CGC结构
        self.cgc_layers = nn.ModuleList()
        curr_input_dim = input_dim
        
        for _ in range(num_layers):
            # 添加CGC层
            cgc_layer = CGCLayer(
                input_dim=curr_input_dim,
                shared_experts=shared_experts,
                task_specific_experts=task_specific_experts,
                expert_dim=expert_dim,
                num_tasks=num_tasks,
                gate_dim=gate_dim
            )
            self.cgc_layers.append(cgc_layer)
            curr_input_dim = expert_dim
        
        # 任务特定的输出层
        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, expert_dim // 2),
                nn.ReLU(),
                nn.Linear(expert_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])
        
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
        # 通过多层CGC结构
        task_outputs = None
        for i, cgc_layer in enumerate(self.cgc_layers):
            # 第一层使用原始输入，后续层使用上一层的输出
            if i == 0:
                task_outputs = cgc_layer(x)
            else:
                task_outputs = cgc_layer(task_outputs)
                
        # 通过任务塔获取最终预测
        final_outputs = []
        for task_id in range(self.num_tasks):
            task_pred = self.task_towers[task_id](task_outputs[task_id])
            final_outputs.append(task_pred)
            
        return final_outputs
    
    def calculate_loss(self, predictions, targets, task_weights=None):
        """
        计算多目标损失
        
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


class CGCLayer(nn.Module):
    """
    Customized Gate Control (CGC) 层
    
    为每个任务提供特定专家网络，同时也有共享专家网络
    """
    def __init__(self, input_dim, shared_experts, task_specific_experts, expert_dim, num_tasks, gate_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.shared_experts = shared_experts
        self.task_specific_experts = task_specific_experts
        self.expert_dim = expert_dim
        self.num_tasks = num_tasks
        
        # 共享专家网络
        self.shared_experts_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, expert_dim),
                nn.ReLU()
            ) for _ in range(shared_experts)
        ])
        
        # 任务特定专家网络
        self.task_experts_nets = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, expert_dim),
                    nn.ReLU(),
                    nn.Linear(expert_dim, expert_dim),
                    nn.ReLU()
                ) for _ in range(task_specific_experts)
            ]) for _ in range(num_tasks)
        ])
        
        # 每个任务的选择门控
        # 为每个任务选择共享专家和任务特定专家
        total_experts_per_task = shared_experts + task_specific_experts
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, gate_dim),
                nn.ReLU(),
                nn.Linear(gate_dim, total_experts_per_task),
                nn.Softmax(dim=1)
            ) for _ in range(num_tasks)
        ])
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 如果是第一层，输入形状为(batch_size, input_dim)
               如果不是第一层，输入为上一层的输出，list of tensors
               
        返回:
            list: 每个任务的输出特征
        """
        batch_size = x.shape[0] if isinstance(x, torch.Tensor) else x[0].shape[0]
        device = x.device if isinstance(x, torch.Tensor) else x[0].device
        
        # 通过共享专家网络
        shared_outputs = []
        for expert in self.shared_experts_nets:
            if isinstance(x, list):
                # 如果输入是上一层的任务输出列表，使用平均池化来聚合
                expert_inputs = torch.stack(x, dim=0).mean(dim=0)
                shared_outputs.append(expert(expert_inputs))
            else:
                # 如果是第一层输入
                shared_outputs.append(expert(x))
        
        # 计算每个任务的输出
        task_outputs = []
        for task_id in range(self.num_tasks):
            # 获取任务特定专家输出
            task_expert_outputs = []
            for expert in self.task_experts_nets[task_id]:
                if isinstance(x, list):
                    # 使用该任务的上一层输出
                    task_expert_outputs.append(expert(x[task_id]))
                else:
                    # 第一层输入
                    task_expert_outputs.append(expert(x))
            
            # 合并共享专家和任务专家的输出
            all_expert_outputs = shared_outputs + task_expert_outputs
            all_expert_outputs = torch.stack(all_expert_outputs, dim=1)  # (batch_size, num_experts, expert_dim)
            
            # 应用任务门控
            if isinstance(x, list):
                gate_input = x[task_id]
            else:
                gate_input = x
            
            gate_values = self.gates[task_id](gate_input).unsqueeze(-1)  # (batch_size, num_experts, 1)
            task_output = torch.sum(all_expert_outputs * gate_values, dim=1)  # (batch_size, expert_dim)
            task_outputs.append(task_output)
            
        return task_outputs


class MultiObjectiveRecommender:
    """
    多目标推荐系统
    
    可以同时优化多个目标，如点击率(CTR)、转化率(CVR)、用户满意度等
    """
    def __init__(self, 
                 input_dim: int,
                 task_names: List[str],
                 model_config: Dict[str, Any] = None):
        """
        初始化多目标推荐系统
        
        参数:
            input_dim (int): 输入特征维度
            task_names (List[str]): 任务名列表，例如 ["ctr", "cvr", "satisfaction"]
            model_config (Dict): 模型配置参数
        """
        self.input_dim = input_dim
        self.task_names = task_names
        self.num_tasks = len(task_names)
        
        # 默认模型配置
        default_config = {
            "shared_experts": 4,
            "task_specific_experts": 3,
            "expert_dim": 64,
            "num_layers": 2,
            "gate_dim": 32,
            "learning_rate": 0.001,
            "batch_size": 256,
            "epochs": 10,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        }
        
        # 更新配置
        self.config = default_config
        if model_config:
            self.config.update(model_config)
            
        # 初始化模型
        self.model = PLE(
            input_dim=input_dim,
            shared_experts=self.config["shared_experts"],
            task_specific_experts=self.config["task_specific_experts"],
            expert_dim=self.config["expert_dim"],
            num_tasks=self.num_tasks,
            num_layers=self.config["num_layers"],
            gate_dim=self.config["gate_dim"]
        )
        
        # 特征预处理器
        self.scaler = StandardScaler()
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config["learning_rate"]
        )
        
    def preprocess_features(self, features_df: pd.DataFrame) -> torch.Tensor:
        """
        预处理输入特征
        
        参数:
            features_df (pd.DataFrame): 特征数据框
            
        返回:
            torch.Tensor: 处理后的特征张量
        """
        features = self.scaler.fit_transform(features_df.values)
        return torch.FloatTensor(features)
    
    def create_dataloader(self, 
                        features: torch.Tensor, 
                        targets: List[torch.Tensor],
                        batch_size: Optional[int] = None) -> torch.utils.data.DataLoader:
        """
        创建PyTorch数据加载器
        
        参数:
            features (torch.Tensor): 特征张量
            targets (List[torch.Tensor]): 每个任务的目标张量列表
            batch_size (int): 批次大小
            
        返回:
            DataLoader: PyTorch数据加载器
        """
        if batch_size is None:
            batch_size = self.config["batch_size"]
            
        dataset = torch.utils.data.TensorDataset(features, *targets)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train(self, 
              features_df: pd.DataFrame, 
              targets_dict: Dict[str, np.ndarray],
              task_weights: Optional[List[float]] = None,
              validation_ratio: float = 0.2,
              epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        训练模型
        
        参数:
            features_df (pd.DataFrame): 特征数据框
            targets_dict (Dict[str, np.ndarray]): 任务名到目标值的映射
            task_weights (List[float]): 任务权重
            validation_ratio (float): 验证集比例
            epochs (int): 训练轮数
            
        返回:
            Dict: 包含训练历史的字典
        """
        if epochs is None:
            epochs = self.config["epochs"]
            
        # 预处理特征
        features = self.preprocess_features(features_df)
        
        # 确保所有任务都有目标值
        targets = []
        for task_name in self.task_names:
            if task_name not in targets_dict:
                raise ValueError(f"目标值中缺少任务 {task_name}")
            targets.append(torch.FloatTensor(targets_dict[task_name]).view(-1, 1))
            
        # 划分训练集和验证集
        dataset_size = len(features)
        val_size = int(validation_ratio * dataset_size)
        train_size = dataset_size - val_size
        
        train_indices = list(range(train_size))
        val_indices = list(range(train_size, dataset_size))
        
        # 创建训练集和验证集
        train_features = features[train_indices]
        train_targets = [target[train_indices] for target in targets]
        
        val_features = features[val_indices]
        val_targets = [target[val_indices] for target in targets]
        
        # 创建数据加载器
        train_loader = self.create_dataloader(train_features, train_targets)
        val_loader = self.create_dataloader(val_features, val_targets)
        
        # 将模型移到指定设备
        device = self.config["device"]
        self.model.to(device)
        
        # 训练历史
        history = {
            "train_loss": [],
            "val_loss": []
        }
        for task_name in self.task_names:
            history[f"train_{task_name}_loss"] = []
            history[f"val_{task_name}_loss"] = []
            
        # 开始训练
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_total_loss = 0
            train_task_losses = [0] * self.num_tasks
            
            for batch in train_loader:
                batch_features = batch[0].to(device)
                batch_targets = [b.to(device) for b in batch[1:]]
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_features)
                
                loss, task_losses = self.model.calculate_loss(predictions, batch_targets, task_weights)
                loss.backward()
                self.optimizer.step()
                
                train_total_loss += loss.item()
                for i, task_loss in enumerate(task_losses):
                    train_task_losses[i] += task_loss
                    
            # 验证阶段
            self.model.eval()
            val_total_loss = 0
            val_task_losses = [0] * self.num_tasks
            
            with torch.no_grad():
                for batch in val_loader:
                    batch_features = batch[0].to(device)
                    batch_targets = [b.to(device) for b in batch[1:]]
                    
                    predictions = self.model(batch_features)
                    loss, task_losses = self.model.calculate_loss(predictions, batch_targets, task_weights)
                    
                    val_total_loss += loss.item()
                    for i, task_loss in enumerate(task_losses):
                        val_task_losses[i] += task_loss
            
            # 记录训练指标
            train_avg_loss = train_total_loss / len(train_loader)
            val_avg_loss = val_total_loss / len(val_loader)
            
            history["train_loss"].append(train_avg_loss)
            history["val_loss"].append(val_avg_loss)
            
            for i, task_name in enumerate(self.task_names):
                train_avg_task_loss = train_task_losses[i] / len(train_loader)
                val_avg_task_loss = val_task_losses[i] / len(val_loader)
                
                history[f"train_{task_name}_loss"].append(train_avg_task_loss)
                history[f"val_{task_name}_loss"].append(val_avg_task_loss)
            
            # 打印训练进度
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_avg_loss:.4f}, Val Loss: {val_avg_loss:.4f}")
            for i, task_name in enumerate(self.task_names):
                print(f"{task_name} - Train: {history[f'train_{task_name}_loss'][-1]:.4f}, "
                      f"Val: {history[f'val_{task_name}_loss'][-1]:.4f}")
            print()
            
        return history
    
    def predict(self, features_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        预测多个目标
        
        参数:
            features_df (pd.DataFrame): 特征数据框
            
        返回:
            Dict[str, np.ndarray]: 任务名到预测结果的映射
        """
        # 预处理特征
        features = self.preprocess_features(features_df)
        device = self.config["device"]
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            features_tensor = features.to(device)
            predictions = self.model(features_tensor)
            predictions = [p.cpu().numpy() for p in predictions]
            
        # 将预测结果映射到任务名
        results = {}
        for i, task_name in enumerate(self.task_names):
            results[task_name] = predictions[i]
            
        return results
    
    def save_model(self, path: str):
        """
        保存模型
        
        参数:
            path (str): 保存路径
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler": self.scaler,
            "config": self.config,
            "task_names": self.task_names,
            "input_dim": self.input_dim
        }
        torch.save(checkpoint, path)
        
    @classmethod
    def load_model(cls, path: str):
        """
        加载模型
        
        参数:
            path (str): 模型路径
            
        返回:
            MultiObjectiveRecommender: 加载的推荐模型
        """
        checkpoint = torch.load(path)
        
        # 创建模型实例
        recommender = cls(
            input_dim=checkpoint["input_dim"],
            task_names=checkpoint["task_names"],
            model_config=checkpoint["config"]
        )
        
        # 加载模型参数和优化器状态
        recommender.model.load_state_dict(checkpoint["model_state_dict"])
        recommender.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        recommender.scaler = checkpoint["scaler"]
        
        return recommender 