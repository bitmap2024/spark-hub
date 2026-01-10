"""
知识库推荐系统 - 召回层（双塔模型）

本模块实现了用于快速候选召回的双塔模型（Two-Tower Model）
- 用户塔：编码用户特征和历史行为
- 知识塔：编码知识内容特征
- 在线服务时，通过向量相似度快速检索候选

特点：
- 亿级数据到千级候选的毫秒级召回
- 支持向量数据库（FAISS/Milvus）加速
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from feature_encoder import (
    UserFeatureEncoder, KnowledgeFeatureEncoder, 
    ContextFeatureEncoder, SequenceEncoder
)


class UserTower(nn.Module):
    """
    用户塔
    
    将用户特征、上下文特征、历史行为序列编码为用户向量
    """
    def __init__(self, config: Dict):
        """
        初始化用户塔
        
        参数:
            config (Dict): 配置参数
        """
        super().__init__()
        
        embedding_dim = config['embedding_dim']
        
        # 用户特征编码器
        self.user_encoder = UserFeatureEncoder(config['user_config'])
        
        # 上下文特征编码器
        self.context_encoder = ContextFeatureEncoder(config['context_config'])
        
        # 历史行为序列编码（使用平均池化，因为召回阶段没有候选物品做注意力）
        self.history_embedding = nn.Embedding(config['num_knowledge'], embedding_dim)
        self.history_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
        
        # 用户向量融合层
        input_dim = self.user_encoder.output_dim + self.context_encoder.output_dim + embedding_dim
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.output_dim = embedding_dim
        
    def forward(self, user_features: Dict, context_features: Dict,
                history_ids: torch.LongTensor, history_lengths: torch.LongTensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            user_features (Dict): 用户特征
            context_features (Dict): 上下文特征
            history_ids (torch.LongTensor): 历史行为知识ID序列
            history_lengths (torch.LongTensor): 历史序列实际长度
            
        返回:
            torch.Tensor: 用户向量，形状为 (batch_size, embedding_dim)
        """
        # 编码用户特征
        user_emb = self.user_encoder(user_features)
        
        # 编码上下文特征
        context_emb = self.context_encoder(context_features)
        
        # 编码历史行为（平均池化）
        history_emb = self.history_embedding(history_ids)  # (batch, seq_len, dim)
        
        # 创建掩码
        batch_size, max_len = history_ids.size()
        mask = torch.arange(max_len, device=history_ids.device).unsqueeze(0).expand(batch_size, -1)
        mask = (mask < history_lengths.unsqueeze(1)).float().unsqueeze(-1)
        
        # 平均池化
        history_emb = (history_emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        history_emb = self.history_encoder(history_emb)
        
        # 融合所有特征
        combined = torch.cat([user_emb, context_emb, history_emb], dim=-1)
        user_vector = self.fusion(combined)
        
        # L2 归一化（用于余弦相似度计算）
        user_vector = F.normalize(user_vector, p=2, dim=-1)
        
        return user_vector


class KnowledgeTower(nn.Module):
    """
    知识塔
    
    将知识内容特征编码为知识向量
    """
    def __init__(self, config: Dict):
        """
        初始化知识塔
        
        参数:
            config (Dict): 配置参数
        """
        super().__init__()
        
        embedding_dim = config['embedding_dim']
        
        # 知识特征编码器
        self.knowledge_encoder = KnowledgeFeatureEncoder(config['knowledge_config'])
        
        # 知识向量投影层
        self.projection = nn.Sequential(
            nn.Linear(self.knowledge_encoder.output_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.output_dim = embedding_dim
        
    def forward(self, knowledge_features: Dict) -> torch.Tensor:
        """
        前向传播
        
        参数:
            knowledge_features (Dict): 知识特征
            
        返回:
            torch.Tensor: 知识向量，形状为 (batch_size, embedding_dim)
        """
        knowledge_emb = self.knowledge_encoder(knowledge_features)
        knowledge_vector = self.projection(knowledge_emb)
        
        # L2 归一化（用于余弦相似度计算）
        knowledge_vector = F.normalize(knowledge_vector, p=2, dim=-1)
        
        return knowledge_vector


class TwoTowerRetrieval(nn.Module):
    """
    双塔召回模型
    
    通过分离的用户塔和知识塔实现高效的候选召回
    """
    def __init__(self, config: Dict):
        """
        初始化双塔模型
        
        参数:
            config (Dict): 配置参数
        """
        super().__init__()
        
        self.user_tower = UserTower(config)
        self.knowledge_tower = KnowledgeTower(config)
        
        # 温度参数（用于softmax）
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward(self, user_features: Dict, context_features: Dict,
                history_ids: torch.LongTensor, history_lengths: torch.LongTensor,
                knowledge_features: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
            user_features (Dict): 用户特征
            context_features (Dict): 上下文特征
            history_ids (torch.LongTensor): 历史行为知识ID
            history_lengths (torch.LongTensor): 历史序列长度
            knowledge_features (Dict): 知识特征
            
        返回:
            Tuple: (用户向量, 知识向量, 相似度分数)
        """
        # 获取用户向量
        user_vector = self.user_tower(user_features, context_features, history_ids, history_lengths)
        
        # 获取知识向量
        knowledge_vector = self.knowledge_tower(knowledge_features)
        
        # 计算相似度（余弦相似度）
        similarity = torch.sum(user_vector * knowledge_vector, dim=-1) / self.temperature
        
        return user_vector, knowledge_vector, similarity
    
    def get_user_embedding(self, user_features: Dict, context_features: Dict,
                           history_ids: torch.LongTensor, 
                           history_lengths: torch.LongTensor) -> torch.Tensor:
        """
        获取用户向量（用于在线服务）
        
        参数:
            user_features (Dict): 用户特征
            context_features (Dict): 上下文特征
            history_ids (torch.LongTensor): 历史行为知识ID
            history_lengths (torch.LongTensor): 历史序列长度
            
        返回:
            torch.Tensor: 用户向量
        """
        return self.user_tower(user_features, context_features, history_ids, history_lengths)
    
    def get_knowledge_embedding(self, knowledge_features: Dict) -> torch.Tensor:
        """
        获取知识向量（用于离线建库）
        
        参数:
            knowledge_features (Dict): 知识特征
            
        返回:
            torch.Tensor: 知识向量
        """
        return self.knowledge_tower(knowledge_features)
    
    def compute_loss(self, user_vector: torch.Tensor, knowledge_vector: torch.Tensor,
                     negative_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算对比学习损失（InfoNCE Loss）
        
        参数:
            user_vector (torch.Tensor): 用户向量，形状为 (batch_size, dim)
            knowledge_vector (torch.Tensor): 正样本知识向量，形状为 (batch_size, dim)
            negative_vectors (torch.Tensor, optional): 负样本向量
            
        返回:
            torch.Tensor: 损失值
        """
        batch_size = user_vector.size(0)
        
        # 如果没有提供负样本，使用batch内负采样
        if negative_vectors is None:
            # 构建相似度矩阵 (batch_size, batch_size)
            similarity_matrix = torch.mm(user_vector, knowledge_vector.t()) / self.temperature
            
            # 对角线是正样本
            labels = torch.arange(batch_size, device=user_vector.device)
            
            # 计算交叉熵损失
            loss = F.cross_entropy(similarity_matrix, labels)
        else:
            # 使用提供的负样本
            # positive: (batch_size, 1)
            pos_score = torch.sum(user_vector * knowledge_vector, dim=-1, keepdim=True) / self.temperature
            
            # negative: (batch_size, num_negatives)
            neg_score = torch.mm(user_vector, negative_vectors.t()) / self.temperature
            
            # 拼接并计算softmax损失
            logits = torch.cat([pos_score, neg_score], dim=-1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=user_vector.device)
            loss = F.cross_entropy(logits, labels)
            
        return loss
    

class HardNegativeMiner:
    """
    困难负样本挖掘器
    
    从召回结果中挖掘困难负样本，提升模型训练效果
    """
    def __init__(self, model: TwoTowerRetrieval, knowledge_vectors: torch.Tensor,
                 num_hard_negatives: int = 10):
        """
        初始化困难负样本挖掘器
        
        参数:
            model (TwoTowerRetrieval): 双塔模型
            knowledge_vectors (torch.Tensor): 所有知识向量
            num_hard_negatives (int): 每个样本的困难负样本数量
        """
        self.model = model
        self.knowledge_vectors = knowledge_vectors
        self.num_hard_negatives = num_hard_negatives
        
    def mine(self, user_vectors: torch.Tensor, positive_ids: torch.LongTensor) -> torch.Tensor:
        """
        挖掘困难负样本
        
        参数:
            user_vectors (torch.Tensor): 用户向量
            positive_ids (torch.LongTensor): 正样本ID
            
        返回:
            torch.Tensor: 困难负样本向量
        """
        batch_size = user_vectors.size(0)
        
        # 计算与所有知识的相似度
        with torch.no_grad():
            similarity = torch.mm(user_vectors, self.knowledge_vectors.t())
            
            # 排除正样本
            for i in range(batch_size):
                similarity[i, positive_ids[i]] = float('-inf')
            
            # 获取top-k相似但不是正样本的知识（困难负样本）
            _, hard_negative_ids = torch.topk(similarity, self.num_hard_negatives, dim=-1)
            
            # 获取困难负样本向量
            hard_negatives = self.knowledge_vectors[hard_negative_ids.view(-1)].view(
                batch_size, self.num_hard_negatives, -1
            )
            
        return hard_negatives


class MultiChannelRetrieval(nn.Module):
    """
    多路召回模型
    
    组合多个召回通道：
    - 双塔召回（基于用户兴趣）
    - 热门召回（基于全局热度）
    - 新内容召回（扶持新发布内容）
    - 协同召回（基于相似用户行为）
    """
    def __init__(self, config: Dict):
        """
        初始化多路召回模型
        
        参数:
            config (Dict): 配置参数
        """
        super().__init__()
        
        self.two_tower = TwoTowerRetrieval(config)
        
        # 召回通道权重
        self.channel_weights = nn.Parameter(torch.tensor([0.5, 0.2, 0.15, 0.15]))
        
    def forward(self, user_features: Dict, context_features: Dict,
                history_ids: torch.LongTensor, history_lengths: torch.LongTensor,
                candidate_features: Dict, 
                hot_scores: torch.Tensor,
                freshness_scores: torch.Tensor,
                cf_scores: torch.Tensor) -> torch.Tensor:
        """
        多路召回融合
        
        参数:
            user_features (Dict): 用户特征
            context_features (Dict): 上下文特征
            history_ids (torch.LongTensor): 历史行为
            history_lengths (torch.LongTensor): 历史长度
            candidate_features (Dict): 候选知识特征
            hot_scores (torch.Tensor): 热门分数
            freshness_scores (torch.Tensor): 新鲜度分数
            cf_scores (torch.Tensor): 协同过滤分数
            
        返回:
            torch.Tensor: 融合后的召回分数
        """
        # 双塔召回分数
        _, _, two_tower_scores = self.two_tower(
            user_features, context_features, history_ids, history_lengths, candidate_features
        )
        
        # 归一化各路分数
        two_tower_scores = torch.sigmoid(two_tower_scores)
        hot_scores = torch.sigmoid(hot_scores)
        freshness_scores = torch.sigmoid(freshness_scores)
        cf_scores = torch.sigmoid(cf_scores)
        
        # 加权融合
        weights = F.softmax(self.channel_weights, dim=0)
        final_scores = (
            weights[0] * two_tower_scores +
            weights[1] * hot_scores +
            weights[2] * freshness_scores +
            weights[3] * cf_scores
        )
        
        return final_scores


def train_two_tower(model: TwoTowerRetrieval, train_loader, 
                    optimizer, num_epochs: int = 10, device: str = 'cuda'):
    """
    训练双塔模型
    
    参数:
        model (TwoTowerRetrieval): 双塔模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        num_epochs (int): 训练轮数
        device (str): 训练设备
        
    返回:
        List: 训练损失历史
    """
    model.to(device)
    history = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # 移动数据到设备
            user_features = {k: v.to(device) for k, v in batch['user_features'].items()}
            context_features = {k: v.to(device) for k, v in batch['context_features'].items()}
            history_ids = batch['history_ids'].to(device)
            history_lengths = batch['history_lengths'].to(device)
            knowledge_features = {k: v.to(device) for k, v in batch['knowledge_features'].items()}
            
            # 前向传播
            user_vector, knowledge_vector, _ = model(
                user_features, context_features, history_ids, history_lengths, knowledge_features
            )
            
            # 计算损失
            loss = model.compute_loss(user_vector, knowledge_vector)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        history.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
    return history


if __name__ == "__main__":
    # 测试配置
    config = {
        'embedding_dim': 64,
        'num_knowledge': 50000,
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
    model = TwoTowerRetrieval(config)
    
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
    
    context_features = {
        'hour': torch.randint(0, 24, (batch_size,)),
        'weekday': torch.randint(0, 7, (batch_size,)),
        'device': torch.randint(0, 10, (batch_size,)),
        'platform': torch.randint(0, 5, (batch_size,))
    }
    
    history_ids = torch.randint(0, 50000, (batch_size, 50))
    history_lengths = torch.randint(1, 50, (batch_size,))
    
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
    
    # 前向传播测试
    user_vec, knowledge_vec, scores = model(
        user_features, context_features, history_ids, history_lengths, knowledge_features
    )
    
    print(f"用户向量维度: {user_vec.shape}")
    print(f"知识向量维度: {knowledge_vec.shape}")
    print(f"相似度分数维度: {scores.shape}")
    
    # 计算损失
    loss = model.compute_loss(user_vec, knowledge_vec)
    print(f"损失值: {loss.item():.4f}")
    
    print("\n✅ 双塔召回模型测试通过！")

