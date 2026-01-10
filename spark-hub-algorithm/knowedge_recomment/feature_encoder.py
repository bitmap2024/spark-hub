"""
知识库推荐系统 - 特征编码模块

本模块实现了针对知识库场景的特征工程，包括：
- 用户特征编码（静态特征 + 动态兴趣）
- 知识内容特征编码（文本 + 元数据）
- 上下文特征编码（时间 + 设备等）
- 序列特征编码（用户历史行为）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class NumericalBucketing(nn.Module):
    """
    数值特征分桶编码
    
    将连续数值特征离散化为桶，然后转换为Embedding
    支持等距分桶和对数分桶，可处理异常值
    """
    def __init__(self, num_buckets: int, embedding_dim: int, use_log: bool = False):
        """
        初始化数值分桶编码器
        
        参数:
            num_buckets (int): 分桶数量
            embedding_dim (int): 嵌入维度
            use_log (bool): 是否使用对数转换（适用于长尾分布）
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.use_log = use_log
        self.embedding = nn.Embedding(num_buckets + 2, embedding_dim)  # +2 for padding and overflow
        nn.init.xavier_normal_(self.embedding.weight)
        
    def forward(self, x: torch.Tensor, min_val: float = 0.0, max_val: float = 100.0) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入数值特征，形状为 (batch_size,)
            min_val (float): 最小值
            max_val (float): 最大值
            
        返回:
            torch.Tensor: 嵌入向量，形状为 (batch_size, embedding_dim)
        """
        if self.use_log:
            # 对数转换，压缩动态范围
            x = torch.log1p(x.clamp(min=0))
            min_val = np.log1p(max(0, min_val))
            max_val = np.log1p(max_val)
        
        # 将数值映射到桶索引
        bucket_idx = ((x - min_val) / (max_val - min_val + 1e-8) * self.num_buckets).long()
        bucket_idx = bucket_idx.clamp(0, self.num_buckets - 1)
        
        return self.embedding(bucket_idx)


class CategoricalEncoder(nn.Module):
    """
    类别特征编码器
    
    将离散类别特征（如：知识类别、难度级别）转换为Embedding
    """
    def __init__(self, num_categories: int, embedding_dim: int, padding_idx: int = 0):
        """
        初始化类别编码器
        
        参数:
            num_categories (int): 类别数量
            embedding_dim (int): 嵌入维度
            padding_idx (int): 填充索引
        """
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim, padding_idx=padding_idx)
        nn.init.xavier_normal_(self.embedding.weight)
        if padding_idx is not None:
            self.embedding.weight.data[padding_idx].zero_()
            
    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x (torch.LongTensor): 类别索引，形状为 (batch_size,)
            
        返回:
            torch.Tensor: 嵌入向量，形状为 (batch_size, embedding_dim)
        """
        return self.embedding(x)


class MultiHotEncoder(nn.Module):
    """
    多热编码器
    
    用于处理多标签特征（如：知识标签、技能标签）
    """
    def __init__(self, num_categories: int, embedding_dim: int, pooling: str = 'mean'):
        """
        初始化多热编码器
        
        参数:
            num_categories (int): 类别数量
            embedding_dim (int): 嵌入维度
            pooling (str): 池化方式 ('mean', 'sum', 'max')
        """
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim, padding_idx=0)
        self.pooling = pooling
        nn.init.xavier_normal_(self.embedding.weight)
        self.embedding.weight.data[0].zero_()
        
    def forward(self, x: torch.LongTensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x (torch.LongTensor): 多标签索引，形状为 (batch_size, max_num_labels)
            mask (torch.Tensor, optional): 有效标签掩码
            
        返回:
            torch.Tensor: 池化后的嵌入向量，形状为 (batch_size, embedding_dim)
        """
        emb = self.embedding(x)  # (batch_size, max_num_labels, embedding_dim)
        
        if mask is None:
            mask = (x != 0).float()
        
        mask = mask.unsqueeze(-1)  # (batch_size, max_num_labels, 1)
        
        if self.pooling == 'mean':
            return (emb * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
        elif self.pooling == 'sum':
            return (emb * mask).sum(dim=1)
        elif self.pooling == 'max':
            emb = emb.masked_fill(mask == 0, float('-inf'))
            return emb.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")


class TextEncoder(nn.Module):
    """
    文本特征编码器
    
    将知识标题、摘要等文本特征编码为向量
    支持预训练语言模型的嵌入
    """
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.1, pretrained_emb: Optional[torch.Tensor] = None):
        """
        初始化文本编码器
        
        参数:
            vocab_size (int): 词表大小
            embedding_dim (int): 词嵌入维度
            hidden_dim (int): 隐藏层维度
            num_layers (int): LSTM层数
            dropout (float): Dropout比率
            pretrained_emb (torch.Tensor, optional): 预训练词嵌入
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
        else:
            nn.init.xavier_normal_(self.embedding.weight)
        self.embedding.weight.data[0].zero_()
        
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_dim = hidden_dim * 2  # 双向LSTM
        
    def forward(self, x: torch.LongTensor, lengths: torch.LongTensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x (torch.LongTensor): token索引，形状为 (batch_size, max_seq_len)
            lengths (torch.LongTensor): 序列实际长度
            
        返回:
            torch.Tensor: 文本编码向量，形状为 (batch_size, hidden_dim * 2)
        """
        emb = self.embedding(x)  # (batch_size, max_seq_len, embedding_dim)
        
        # 打包变长序列
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        
        _, (hidden, _) = self.lstm(packed)
        
        # 连接前向和后向的最后隐藏状态
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch_size, hidden_dim * 2)
        
        return hidden


class UserFeatureEncoder(nn.Module):
    """
    用户特征编码器
    
    整合用户的静态特征和动态特征
    """
    def __init__(self, config: Dict):
        """
        初始化用户特征编码器
        
        参数:
            config (Dict): 配置参数
                - num_users: 用户数量
                - num_levels: 用户等级数量
                - num_professions: 职业数量
                - num_interest_tags: 兴趣标签数量
                - embedding_dim: 嵌入维度
        """
        super().__init__()
        
        embedding_dim = config['embedding_dim']
        
        # 用户ID嵌入
        self.user_embedding = nn.Embedding(config['num_users'], embedding_dim)
        
        # 用户等级编码（学习者等级：初学者、进阶、专家等）
        self.level_encoder = CategoricalEncoder(config['num_levels'], embedding_dim)
        
        # 职业/领域编码
        self.profession_encoder = CategoricalEncoder(config['num_professions'], embedding_dim)
        
        # 兴趣标签编码（多标签）
        self.interest_encoder = MultiHotEncoder(config['num_interest_tags'], embedding_dim)
        
        # 数值特征：学习时长（天）、完成率等
        self.learning_days_encoder = NumericalBucketing(20, embedding_dim, use_log=True)
        self.completion_rate_encoder = NumericalBucketing(10, embedding_dim)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 6, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.output_dim = embedding_dim
        
    def forward(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        参数:
            user_features (Dict): 用户特征字典
                - user_id: 用户ID
                - level: 用户等级
                - profession: 职业
                - interest_tags: 兴趣标签
                - learning_days: 学习天数
                - completion_rate: 完成率
                
        返回:
            torch.Tensor: 用户嵌入向量
        """
        user_emb = self.user_embedding(user_features['user_id'])
        level_emb = self.level_encoder(user_features['level'])
        profession_emb = self.profession_encoder(user_features['profession'])
        interest_emb = self.interest_encoder(user_features['interest_tags'])
        learning_days_emb = self.learning_days_encoder(user_features['learning_days'].float(), 0, 1000)
        completion_rate_emb = self.completion_rate_encoder(user_features['completion_rate'].float(), 0, 100)
        
        # 拼接所有特征
        combined = torch.cat([
            user_emb, level_emb, profession_emb, 
            interest_emb, learning_days_emb, completion_rate_emb
        ], dim=-1)
        
        return self.fusion(combined)


class KnowledgeFeatureEncoder(nn.Module):
    """
    知识内容特征编码器
    
    编码知识库中内容的各类特征
    """
    def __init__(self, config: Dict):
        """
        初始化知识特征编码器
        
        参数:
            config (Dict): 配置参数
                - num_knowledge: 知识内容数量
                - num_categories: 知识类别数量
                - num_difficulty_levels: 难度级别数量
                - num_content_tags: 内容标签数量
                - vocab_size: 词表大小
                - embedding_dim: 嵌入维度
        """
        super().__init__()
        
        embedding_dim = config['embedding_dim']
        
        # 知识ID嵌入
        self.knowledge_embedding = nn.Embedding(config['num_knowledge'], embedding_dim)
        
        # 类别编码（技术、管理、设计等）
        self.category_encoder = CategoricalEncoder(config['num_categories'], embedding_dim)
        
        # 难度级别编码（入门、初级、中级、高级、专家）
        self.difficulty_encoder = CategoricalEncoder(config['num_difficulty_levels'], embedding_dim)
        
        # 内容标签编码（多标签）
        self.tag_encoder = MultiHotEncoder(config['num_content_tags'], embedding_dim)
        
        # 标题文本编码
        self.title_encoder = TextEncoder(
            config['vocab_size'], embedding_dim // 2, 
            hidden_dim=embedding_dim // 2, num_layers=1
        )
        
        # 数值特征：学习人数、平均评分、预估时长
        self.learner_count_encoder = NumericalBucketing(20, embedding_dim, use_log=True)
        self.rating_encoder = NumericalBucketing(10, embedding_dim)
        self.duration_encoder = NumericalBucketing(15, embedding_dim, use_log=True)
        
        # 发布时间新鲜度编码（距今天数）
        self.freshness_encoder = NumericalBucketing(30, embedding_dim, use_log=True)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 8 + self.title_encoder.output_dim, embedding_dim * 2),
            nn.LayerNorm(embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        self.output_dim = embedding_dim
        
    def forward(self, knowledge_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        参数:
            knowledge_features (Dict): 知识特征字典
                - knowledge_id: 知识ID
                - category: 类别
                - difficulty: 难度级别
                - tags: 内容标签
                - title_tokens: 标题token
                - title_lengths: 标题长度
                - learner_count: 学习人数
                - rating: 平均评分
                - duration: 预估学习时长
                - days_since_publish: 发布天数
                
        返回:
            torch.Tensor: 知识内容嵌入向量
        """
        knowledge_emb = self.knowledge_embedding(knowledge_features['knowledge_id'])
        category_emb = self.category_encoder(knowledge_features['category'])
        difficulty_emb = self.difficulty_encoder(knowledge_features['difficulty'])
        tag_emb = self.tag_encoder(knowledge_features['tags'])
        title_emb = self.title_encoder(
            knowledge_features['title_tokens'], 
            knowledge_features['title_lengths']
        )
        learner_count_emb = self.learner_count_encoder(
            knowledge_features['learner_count'].float(), 0, 100000
        )
        rating_emb = self.rating_encoder(knowledge_features['rating'].float(), 0, 5)
        duration_emb = self.duration_encoder(knowledge_features['duration'].float(), 0, 600)
        freshness_emb = self.freshness_encoder(
            knowledge_features['days_since_publish'].float(), 0, 365
        )
        
        # 拼接所有特征
        combined = torch.cat([
            knowledge_emb, category_emb, difficulty_emb, tag_emb,
            title_emb, learner_count_emb, rating_emb, duration_emb, freshness_emb
        ], dim=-1)
        
        return self.fusion(combined)


class ContextFeatureEncoder(nn.Module):
    """
    上下文特征编码器
    
    编码时间、设备、场景等上下文信息
    """
    def __init__(self, config: Dict):
        """
        初始化上下文特征编码器
        
        参数:
            config (Dict): 配置参数
                - num_hours: 小时数（24）
                - num_weekdays: 星期数（7）
                - num_devices: 设备类型数量
                - num_platforms: 平台数量
                - embedding_dim: 嵌入维度
        """
        super().__init__()
        
        embedding_dim = config['embedding_dim']
        
        # 时间特征：小时、星期几
        self.hour_encoder = CategoricalEncoder(config.get('num_hours', 24), embedding_dim)
        self.weekday_encoder = CategoricalEncoder(config.get('num_weekdays', 7), embedding_dim)
        
        # 设备和平台
        self.device_encoder = CategoricalEncoder(config.get('num_devices', 10), embedding_dim)
        self.platform_encoder = CategoricalEncoder(config.get('num_platforms', 5), embedding_dim)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )
        
        self.output_dim = embedding_dim
        
    def forward(self, context_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        参数:
            context_features (Dict): 上下文特征字典
                - hour: 当前小时
                - weekday: 星期几
                - device: 设备类型
                - platform: 平台
                
        返回:
            torch.Tensor: 上下文嵌入向量
        """
        hour_emb = self.hour_encoder(context_features['hour'])
        weekday_emb = self.weekday_encoder(context_features['weekday'])
        device_emb = self.device_encoder(context_features['device'])
        platform_emb = self.platform_encoder(context_features['platform'])
        
        combined = torch.cat([hour_emb, weekday_emb, device_emb, platform_emb], dim=-1)
        return self.fusion(combined)


class SequenceEncoder(nn.Module):
    """
    用户行为序列编码器
    
    使用Target Attention (DIN风格) 对用户历史行为进行编码
    根据候选知识内容，动态计算用户兴趣表示
    """
    def __init__(self, embedding_dim: int, attention_dim: int = 64, max_seq_len: int = 50):
        """
        初始化序列编码器
        
        参数:
            embedding_dim (int): 嵌入维度
            attention_dim (int): 注意力层维度
            max_seq_len (int): 最大序列长度
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        
        # 注意力计算网络
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim * 4, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1)
        )
        
        # 位置编码
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        
        self.output_dim = embedding_dim
        
    def forward(self, query: torch.Tensor, keys: torch.Tensor, 
                keys_length: torch.LongTensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            query (torch.Tensor): 候选知识嵌入，形状为 (batch_size, embedding_dim)
            keys (torch.Tensor): 历史行为知识嵌入，形状为 (batch_size, max_seq_len, embedding_dim)
            keys_length (torch.LongTensor): 历史序列的实际长度
            
        返回:
            torch.Tensor: 注意力加权后的用户兴趣表示
        """
        batch_size, max_len, _ = keys.size()
        
        # 添加位置编码
        positions = torch.arange(max_len, device=keys.device).unsqueeze(0).expand(batch_size, -1)
        keys = keys + self.position_embedding(positions)
        
        # 扩展query以匹配keys的维度
        query_expanded = query.unsqueeze(1).expand(-1, max_len, -1)
        
        # 计算注意力特征：[q, k, q-k, q*k]
        attention_input = torch.cat([
            query_expanded,
            keys,
            query_expanded - keys,
            query_expanded * keys
        ], dim=-1)
        
        # 计算注意力分数
        attention_scores = self.attention(attention_input).squeeze(-1)  # (batch_size, max_len)
        
        # 创建掩码
        mask = torch.arange(max_len, device=keys_length.device).unsqueeze(0).expand(batch_size, -1)
        mask = (mask < keys_length.unsqueeze(1)).float()
        
        # 应用掩码并归一化
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 计算加权和
        output = torch.bmm(attention_weights.unsqueeze(1), keys).squeeze(1)
        
        return output


class FeatureInteraction(nn.Module):
    """
    特征交叉层 (DCN风格)
    
    显式地构建高阶特征交叉
    """
    def __init__(self, input_dim: int, num_cross_layers: int = 3):
        """
        初始化特征交叉层
        
        参数:
            input_dim (int): 输入维度
            num_cross_layers (int): 交叉层数量
        """
        super().__init__()
        
        self.num_cross_layers = num_cross_layers
        
        # 交叉层权重
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1) * 0.01)
            for _ in range(num_cross_layers)
        ])
        
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_cross_layers)
        ])
        
        self.output_dim = input_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征，形状为 (batch_size, input_dim)
            
        返回:
            torch.Tensor: 交叉后的特征
        """
        x0 = x
        xi = x
        
        for i in range(self.num_cross_layers):
            # x_{i+1} = x_0 * (x_i^T * w_i) + b_i + x_i
            xi_w = torch.matmul(xi, self.cross_weights[i])  # (batch_size, 1)
            xi = x0 * xi_w + self.cross_biases[i] + xi
            
        return xi


if __name__ == "__main__":
    # 测试代码
    batch_size = 32
    embedding_dim = 64
    
    # 测试用户特征编码器
    user_config = {
        'num_users': 10000,
        'num_levels': 5,
        'num_professions': 20,
        'num_interest_tags': 100,
        'embedding_dim': embedding_dim
    }
    
    user_encoder = UserFeatureEncoder(user_config)
    
    user_features = {
        'user_id': torch.randint(0, 10000, (batch_size,)),
        'level': torch.randint(0, 5, (batch_size,)),
        'profession': torch.randint(0, 20, (batch_size,)),
        'interest_tags': torch.randint(0, 100, (batch_size, 10)),
        'learning_days': torch.randint(0, 1000, (batch_size,)),
        'completion_rate': torch.randint(0, 100, (batch_size,))
    }
    
    user_emb = user_encoder(user_features)
    print(f"用户特征编码输出维度: {user_emb.shape}")  # (batch_size, embedding_dim)
    
    # 测试知识特征编码器
    knowledge_config = {
        'num_knowledge': 50000,
        'num_categories': 30,
        'num_difficulty_levels': 5,
        'num_content_tags': 200,
        'vocab_size': 30000,
        'embedding_dim': embedding_dim
    }
    
    knowledge_encoder = KnowledgeFeatureEncoder(knowledge_config)
    
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
    
    knowledge_emb = knowledge_encoder(knowledge_features)
    print(f"知识特征编码输出维度: {knowledge_emb.shape}")  # (batch_size, embedding_dim)
    
    # 测试序列编码器
    seq_encoder = SequenceEncoder(embedding_dim)
    history_emb = torch.randn(batch_size, 50, embedding_dim)
    history_lengths = torch.randint(1, 50, (batch_size,))
    
    interest_emb = seq_encoder(knowledge_emb, history_emb, history_lengths)
    print(f"序列编码输出维度: {interest_emb.shape}")  # (batch_size, embedding_dim)
    
    print("\n✅ 特征编码模块测试通过！")

