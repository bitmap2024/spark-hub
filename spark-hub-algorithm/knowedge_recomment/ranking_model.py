"""
知识库推荐系统 - 精排层（PLE多任务学习模型）

本模块实现了用于精细排序的PLE (Progressive Layered Extraction) 模型
- 多任务学习：同时预测点击、收藏、学习完成、分享等多个目标
- 共享专家与任务专属专家分离，解决任务冲突
- Target Attention：根据候选知识动态计算用户兴趣

特点：
- 从几千个候选中精选几百个高质量内容
- 多目标加权融合产生最终排序分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from feature_encoder import (
    UserFeatureEncoder, KnowledgeFeatureEncoder, 
    ContextFeatureEncoder, SequenceEncoder, FeatureInteraction
)


class Expert(nn.Module):
    """
    专家网络
    
    每个专家是一个MLP，学习特定模式的特征表示
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        """
        初始化专家网络
        
        参数:
            input_dim (int): 输入维度
            output_dim (int): 输出维度
            hidden_dim (int): 隐藏层维度
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GatingNetwork(nn.Module):
    """
    门控网络
    
    根据输入特征动态分配各专家的权重
    """
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int = 64):
        """
        初始化门控网络
        
        参数:
            input_dim (int): 输入维度
            num_experts (int): 专家数量
            hidden_dim (int): 隐藏层维度
        """
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


class ExtractionNetwork(nn.Module):
    """
    PLE的提取网络
    
    包含共享专家和任务专属专家
    """
    def __init__(self, input_dim: int, expert_dim: int, 
                 num_shared_experts: int, num_task_experts: int, num_tasks: int):
        """
        初始化提取网络
        
        参数:
            input_dim (int): 输入维度
            expert_dim (int): 专家输出维度
            num_shared_experts (int): 共享专家数量
            num_task_experts (int): 每个任务的专属专家数量
            num_tasks (int): 任务数量
        """
        super().__init__()
        
        self.num_tasks = num_tasks
        self.num_shared_experts = num_shared_experts
        self.num_task_experts = num_task_experts
        
        # 共享专家
        self.shared_experts = nn.ModuleList([
            Expert(input_dim, expert_dim)
            for _ in range(num_shared_experts)
        ])
        
        # 任务专属专家
        self.task_experts = nn.ModuleList([
            nn.ModuleList([
                Expert(input_dim, expert_dim)
                for _ in range(num_task_experts)
            ])
            for _ in range(num_tasks)
        ])
        
        # 任务门控网络
        total_experts = num_shared_experts + num_task_experts
        self.task_gates = nn.ModuleList([
            GatingNetwork(input_dim, total_experts)
            for _ in range(num_tasks)
        ])
        
        # 共享层的门控
        self.shared_gate = GatingNetwork(input_dim, num_shared_experts + num_tasks * num_task_experts)
        
        self.output_dim = expert_dim
        
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征
            
        返回:
            Tuple: (各任务的专家输出列表, 共享层输出)
        """
        batch_size = x.size(0)
        
        # 计算共享专家输出
        shared_outputs = [expert(x) for expert in self.shared_experts]
        shared_outputs_stacked = torch.stack(shared_outputs, dim=1)  # (batch, num_shared, dim)
        
        # 计算任务专属专家输出
        task_outputs_list = []
        for task_id in range(self.num_tasks):
            task_expert_outputs = [expert(x) for expert in self.task_experts[task_id]]
            task_expert_outputs_stacked = torch.stack(task_expert_outputs, dim=1)
            task_outputs_list.append(task_expert_outputs_stacked)
        
        # 为每个任务计算门控加权输出
        task_final_outputs = []
        for task_id in range(self.num_tasks):
            # 拼接共享专家和任务专属专家输出
            combined = torch.cat([shared_outputs_stacked, task_outputs_list[task_id]], dim=1)
            
            # 门控加权
            gate_weights = self.task_gates[task_id](x).unsqueeze(-1)  # (batch, num_experts, 1)
            task_output = (combined * gate_weights).sum(dim=1)  # (batch, dim)
            task_final_outputs.append(task_output)
        
        # 计算共享层输出（用于下一层）
        all_expert_outputs = [shared_outputs_stacked] + task_outputs_list
        all_stacked = torch.cat(all_expert_outputs, dim=1)
        shared_gate_weights = self.shared_gate(x).unsqueeze(-1)
        shared_output = (all_stacked * shared_gate_weights).sum(dim=1)
        
        return task_final_outputs, shared_output


class PLERankingModel(nn.Module):
    """
    PLE精排模型
    
    Progressive Layered Extraction 多任务学习模型
    用于知识库推荐的精细排序
    """
    def __init__(self, config: Dict):
        """
        初始化PLE模型
        
        参数:
            config (Dict): 配置参数
                - embedding_dim: 嵌入维度
                - expert_dim: 专家输出维度
                - num_extraction_layers: 提取层数量
                - num_shared_experts: 共享专家数量
                - num_task_experts: 每个任务的专属专家数量
                - task_names: 任务名称列表
        """
        super().__init__()
        
        embedding_dim = config['embedding_dim']
        expert_dim = config.get('expert_dim', 128)
        num_extraction_layers = config.get('num_extraction_layers', 2)
        num_shared_experts = config.get('num_shared_experts', 3)
        num_task_experts = config.get('num_task_experts', 2)
        
        # 任务定义（知识库推荐场景）
        self.task_names = config.get('task_names', [
            'click',        # 点击率
            'collect',      # 收藏率
            'complete',     # 学习完成率
            'share',        # 分享率
            'duration'      # 停留时长
        ])
        self.num_tasks = len(self.task_names)
        
        # 特征编码器
        self.user_encoder = UserFeatureEncoder(config['user_config'])
        self.knowledge_encoder = KnowledgeFeatureEncoder(config['knowledge_config'])
        self.context_encoder = ContextFeatureEncoder(config['context_config'])
        
        # 历史行为序列编码（Target Attention）
        self.history_embedding = nn.Embedding(config['num_knowledge'], embedding_dim)
        self.sequence_encoder = SequenceEncoder(embedding_dim, max_seq_len=config.get('max_history_len', 50))
        
        # 特征交叉层
        total_feature_dim = (
            self.user_encoder.output_dim +
            self.knowledge_encoder.output_dim +
            self.context_encoder.output_dim +
            self.sequence_encoder.output_dim
        )
        self.feature_interaction = FeatureInteraction(total_feature_dim, num_cross_layers=3)
        
        # PLE提取层
        self.extraction_layers = nn.ModuleList()
        current_dim = total_feature_dim
        
        for i in range(num_extraction_layers):
            layer = ExtractionNetwork(
                input_dim=current_dim,
                expert_dim=expert_dim,
                num_shared_experts=num_shared_experts,
                num_task_experts=num_task_experts,
                num_tasks=self.num_tasks
            )
            self.extraction_layers.append(layer)
            current_dim = expert_dim
        
        # 任务塔
        self.task_towers = nn.ModuleDict()
        for task_name in self.task_names:
            if task_name == 'duration':
                # 停留时长是回归任务
                output_dim = 1
                activation = nn.Identity()
            else:
                # 其他是分类任务
                output_dim = 1
                activation = nn.Sigmoid()
            
            self.task_towers[task_name] = nn.Sequential(
                nn.Linear(expert_dim, expert_dim // 2),
                nn.LayerNorm(expert_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(expert_dim // 2, output_dim),
                activation
            )
        
        # 任务权重（用于最终分数融合）
        self.task_weights = nn.Parameter(torch.ones(self.num_tasks) / self.num_tasks)
        
    def forward(self, user_features: Dict, knowledge_features: Dict,
                context_features: Dict, history_ids: torch.LongTensor,
                history_lengths: torch.LongTensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        参数:
            user_features (Dict): 用户特征
            knowledge_features (Dict): 知识特征
            context_features (Dict): 上下文特征
            history_ids (torch.LongTensor): 历史行为知识ID
            history_lengths (torch.LongTensor): 历史序列长度
            
        返回:
            Dict: 各任务的预测结果
        """
        # 编码各类特征
        user_emb = self.user_encoder(user_features)
        knowledge_emb = self.knowledge_encoder(knowledge_features)
        context_emb = self.context_encoder(context_features)
        
        # 历史序列 Target Attention
        history_emb = self.history_embedding(history_ids)
        interest_emb = self.sequence_encoder(knowledge_emb, history_emb, history_lengths)
        
        # 拼接所有特征
        combined_features = torch.cat([
            user_emb, knowledge_emb, context_emb, interest_emb
        ], dim=-1)
        
        # 特征交叉
        cross_features = self.feature_interaction(combined_features)
        
        # PLE提取
        x = cross_features
        task_outputs = None
        
        for layer in self.extraction_layers:
            task_outputs, x = layer(x)
        
        # 任务塔输出
        predictions = {}
        for i, task_name in enumerate(self.task_names):
            predictions[task_name] = self.task_towers[task_name](task_outputs[i]).squeeze(-1)
        
        return predictions
    
    def compute_loss(self, predictions: Dict[str, torch.Tensor],
                     targets: Dict[str, torch.Tensor],
                     sample_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算多任务损失
        
        参数:
            predictions (Dict): 各任务的预测结果
            targets (Dict): 各任务的目标值
            sample_weights (torch.Tensor, optional): 样本权重
            
        返回:
            Tuple: (总损失, 各任务损失字典)
        """
        task_losses = {}
        total_loss = 0
        
        # 归一化任务权重
        weights = F.softmax(self.task_weights, dim=0)
        
        for i, task_name in enumerate(self.task_names):
            if task_name not in targets:
                continue
                
            pred = predictions[task_name]
            target = targets[task_name]
            
            if task_name == 'duration':
                # 回归任务使用MSE损失
                loss = F.mse_loss(pred, target, reduction='none')
            else:
                # 分类任务使用BCE损失
                loss = F.binary_cross_entropy(pred, target, reduction='none')
            
            # 应用样本权重
            if sample_weights is not None:
                loss = loss * sample_weights
            
            loss = loss.mean()
            task_losses[task_name] = loss.item()
            total_loss += weights[i] * loss
        
        return total_loss, task_losses
    
    def compute_ranking_score(self, predictions: Dict[str, torch.Tensor],
                               score_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        计算排序分数
        
        将多任务预测结果加权融合为最终排序分数
        
        参数:
            predictions (Dict): 各任务的预测结果
            score_weights (Dict, optional): 任务分数权重
            
        返回:
            torch.Tensor: 最终排序分数
        """
        if score_weights is None:
            # 默认权重（知识库场景：更重视学习完成和收藏）
            score_weights = {
                'click': 0.15,
                'collect': 0.30,
                'complete': 0.30,
                'share': 0.10,
                'duration': 0.15
            }
        
        scores = None
        for task_name, weight in score_weights.items():
            if task_name in predictions:
                task_score = predictions[task_name]
                
                # 对duration进行归一化
                if task_name == 'duration':
                    task_score = torch.sigmoid(task_score / 100)  # 假设时长在0-600分钟
                
                if scores is None:
                    scores = weight * task_score
                else:
                    scores = scores + weight * task_score
        
        return scores


class TargetAttentionDIN(nn.Module):
    """
    DIN风格的Target Attention模块
    
    针对不同的候选知识，动态计算用户的实时兴趣
    """
    def __init__(self, embedding_dim: int, attention_dim: int = 64, max_seq_len: int = 50):
        """
        初始化Target Attention模块
        
        参数:
            embedding_dim (int): 嵌入维度
            attention_dim (int): 注意力层维度
            max_seq_len (int): 最大序列长度
        """
        super().__init__()
        
        self.max_seq_len = max_seq_len
        
        # 多头注意力
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 激活单元（类似原始DIN）
        self.activation_unit = nn.Sequential(
            nn.Linear(embedding_dim * 4, attention_dim),
            nn.PReLU(),
            nn.Linear(attention_dim, attention_dim),
            nn.PReLU(),
            nn.Linear(attention_dim, 1)
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.PReLU()
        )
        
        self.output_dim = embedding_dim
        
    def forward(self, query: torch.Tensor, history: torch.Tensor,
                history_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query (torch.Tensor): 候选知识嵌入，形状为 (batch, embedding_dim)
            history (torch.Tensor): 历史行为嵌入，形状为 (batch, seq_len, embedding_dim)
            history_mask (torch.Tensor): 历史序列掩码
            
        返回:
            torch.Tensor: 用户实时兴趣向量
        """
        batch_size, seq_len, dim = history.size()
        
        # 扩展query
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 计算激活单元注意力分数
        activation_input = torch.cat([
            query_expanded,
            history,
            query_expanded - history,
            query_expanded * history
        ], dim=-1)
        
        activation_scores = self.activation_unit(activation_input).squeeze(-1)  # (batch, seq_len)
        
        # 应用掩码
        activation_scores = activation_scores.masked_fill(~history_mask.bool(), float('-inf'))
        attention_weights = F.softmax(activation_scores, dim=-1)
        
        # 加权求和
        activated_interest = torch.bmm(attention_weights.unsqueeze(1), history).squeeze(1)
        
        # 结合多头注意力的结果
        query_for_mha = query.unsqueeze(1)
        mha_output, _ = self.multihead_attention(
            query_for_mha, history, history,
            key_padding_mask=~history_mask.bool()
        )
        mha_output = mha_output.squeeze(1)
        
        # 融合两种注意力结果
        combined = torch.cat([activated_interest, mha_output], dim=-1)
        output = self.output(combined)
        
        return output


def train_ple_model(model: PLERankingModel, train_loader,
                    optimizer, num_epochs: int = 10,
                    device: str = 'cuda', grad_clip: float = 1.0):
    """
    训练PLE精排模型
    
    参数:
        model (PLERankingModel): PLE模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        num_epochs (int): 训练轮数
        device (str): 训练设备
        grad_clip (float): 梯度裁剪阈值
        
    返回:
        Dict: 训练历史
    """
    model.to(device)
    history = {task: [] for task in model.task_names}
    history['total'] = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        task_losses_epoch = {task: 0 for task in model.task_names}
        num_batches = 0
        
        for batch in train_loader:
            # 移动数据到设备
            user_features = {k: v.to(device) for k, v in batch['user_features'].items()}
            knowledge_features = {k: v.to(device) for k, v in batch['knowledge_features'].items()}
            context_features = {k: v.to(device) for k, v in batch['context_features'].items()}
            history_ids = batch['history_ids'].to(device)
            history_lengths = batch['history_lengths'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            
            # 前向传播
            predictions = model(
                user_features, knowledge_features, context_features,
                history_ids, history_lengths
            )
            
            # 计算损失
            loss, task_losses = model.compute_loss(predictions, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            
            total_loss += loss.item()
            for task, task_loss in task_losses.items():
                task_losses_epoch[task] += task_loss
            num_batches += 1
        
        # 记录损失
        avg_total_loss = total_loss / num_batches
        history['total'].append(avg_total_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Total Loss: {avg_total_loss:.4f}")
        
        for task in model.task_names:
            avg_task_loss = task_losses_epoch[task] / num_batches
            history[task].append(avg_task_loss)
            print(f"  {task} Loss: {avg_task_loss:.4f}")
        print()
    
    return history


if __name__ == "__main__":
    # 测试配置
    config = {
        'embedding_dim': 64,
        'expert_dim': 128,
        'num_extraction_layers': 2,
        'num_shared_experts': 3,
        'num_task_experts': 2,
        'num_knowledge': 50000,
        'max_history_len': 50,
        'task_names': ['click', 'collect', 'complete', 'share', 'duration'],
        'user_config': {
            'num_users': 10000,
            'num_levels': 5,
            'num_professions': 20,
            'num_interest_tags': 100,
            'embedding_dim': 64
        },
        'context_config': {
            'num_hours': 24,
            'num_weekdays': 7,
            'num_devices': 10,
            'num_platforms': 5,
            'embedding_dim': 64
        },
        'knowledge_config': {
            'num_knowledge': 50000,
            'num_categories': 30,
            'num_difficulty_levels': 5,
            'num_content_tags': 200,
            'vocab_size': 30000,
            'embedding_dim': 64
        }
    }
    
    # 创建模型
    model = PLERankingModel(config)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟输入数据
    batch_size = 32
    
    user_features = {
        'user_id': torch.randint(0, 10000, (batch_size,)),
        'level': torch.randint(0, 5, (batch_size,)),
        'profession': torch.randint(0, 20, (batch_size,)),
        'interest_tags': torch.randint(0, 100, (batch_size, 10)),
        'learning_days': torch.randint(0, 1000, (batch_size,)),
        'completion_rate': torch.randint(0, 100, (batch_size,))
    }
    
    knowledge_features = {
        'knowledge_id': torch.randint(0, 50000, (batch_size,)),
        'category': torch.randint(0, 30, (batch_size,)),
        'difficulty': torch.randint(0, 5, (batch_size,)),
        'tags': torch.randint(0, 200, (batch_size, 5)),
        'title_tokens': torch.randint(0, 30000, (batch_size, 20)),
        'title_lengths': torch.randint(1, 20, (batch_size,)),
        'learner_count': torch.randint(0, 100000, (batch_size,)),
        'rating': torch.rand(batch_size) * 5,
        'duration': torch.randint(0, 600, (batch_size,)),
        'days_since_publish': torch.randint(0, 365, (batch_size,))
    }
    
    context_features = {
        'hour': torch.randint(0, 24, (batch_size,)),
        'weekday': torch.randint(0, 7, (batch_size,)),
        'device': torch.randint(0, 10, (batch_size,)),
        'platform': torch.randint(0, 5, (batch_size,))
    }
    
    history_ids = torch.randint(0, 50000, (batch_size, 50))
    history_lengths = torch.randint(1, 50, (batch_size,))
    
    # 前向传播测试
    predictions = model(user_features, knowledge_features, context_features, history_ids, history_lengths)
    
    print("\n各任务预测结果:")
    for task_name, pred in predictions.items():
        print(f"  {task_name}: shape={pred.shape}, mean={pred.mean().item():.4f}")
    
    # 计算排序分数
    ranking_scores = model.compute_ranking_score(predictions)
    print(f"\n排序分数: shape={ranking_scores.shape}, mean={ranking_scores.mean().item():.4f}")
    
    # 计算损失
    targets = {
        'click': torch.rand(batch_size),
        'collect': torch.rand(batch_size),
        'complete': torch.rand(batch_size),
        'share': torch.rand(batch_size),
        'duration': torch.rand(batch_size) * 600
    }
    
    loss, task_losses = model.compute_loss(predictions, targets)
    print(f"\n总损失: {loss.item():.4f}")
    for task, task_loss in task_losses.items():
        print(f"  {task} 损失: {task_loss:.4f}")
    
    print("\n✅ PLE精排模型测试通过！")

