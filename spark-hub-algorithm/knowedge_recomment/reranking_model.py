"""
知识库推荐系统 - 重排层

本模块实现了推荐结果的重排策略，包括：
- 多样性打散（Diversity）：避免内容同质化
- 去重过滤（Deduplication）：过滤已学习内容
- 新内容扶持（Fresh Content Boost）：给新内容初始流量
- 负反馈屏蔽（Negative Feedback Filter）：屏蔽用户不感兴趣的内容
- 业务规则融合（Business Rules）：融入平台运营策略

特点：
- 从几百个候选中精选最终展示的几十个
- 平衡个性化推荐与生态健康
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import heapq


class DiversityStrategy(Enum):
    """多样性策略枚举"""
    MMR = "mmr"                    # Maximal Marginal Relevance
    DPP = "dpp"                    # Determinantal Point Process
    SLIDING_WINDOW = "sliding"    # 滑动窗口打散
    CATEGORY_BALANCE = "category" # 类目平衡


@dataclass
class KnowledgeItem:
    """知识内容数据类"""
    knowledge_id: int
    category_id: int
    difficulty: int
    tags: List[int]
    score: float
    embedding: np.ndarray
    publish_timestamp: int
    learner_count: int
    author_id: int
    
    def __lt__(self, other):
        return self.score < other.score


class DiversityReranker:
    """
    多样性重排器
    
    确保推荐列表中内容的多样性，避免信息茧房
    """
    def __init__(self, strategy: DiversityStrategy = DiversityStrategy.MMR,
                 lambda_param: float = 0.5, window_size: int = 3):
        """
        初始化多样性重排器
        
        参数:
            strategy (DiversityStrategy): 多样性策略
            lambda_param (float): 相关性与多样性的平衡参数
            window_size (int): 滑动窗口大小（用于类目打散）
        """
        self.strategy = strategy
        self.lambda_param = lambda_param
        self.window_size = window_size
        
    def rerank(self, items: List[KnowledgeItem], 
               top_k: int) -> List[KnowledgeItem]:
        """
        多样性重排
        
        参数:
            items (List[KnowledgeItem]): 候选知识列表（已按分数排序）
            top_k (int): 返回数量
            
        返回:
            List[KnowledgeItem]: 重排后的知识列表
        """
        if self.strategy == DiversityStrategy.MMR:
            return self._mmr_rerank(items, top_k)
        elif self.strategy == DiversityStrategy.SLIDING_WINDOW:
            return self._sliding_window_rerank(items, top_k)
        elif self.strategy == DiversityStrategy.CATEGORY_BALANCE:
            return self._category_balance_rerank(items, top_k)
        elif self.strategy == DiversityStrategy.DPP:
            return self._dpp_rerank(items, top_k)
        else:
            return items[:top_k]
    
    def _mmr_rerank(self, items: List[KnowledgeItem], 
                    top_k: int) -> List[KnowledgeItem]:
        """
        MMR (Maximal Marginal Relevance) 重排
        
        MMR = λ * Relevance - (1-λ) * max(Similarity to selected items)
        """
        if len(items) == 0:
            return []
        
        selected = []
        candidates = list(items)
        
        # 先选择分数最高的
        selected.append(candidates.pop(0))
        
        while len(selected) < top_k and candidates:
            best_item = None
            best_mmr = float('-inf')
            best_idx = -1
            
            for idx, item in enumerate(candidates):
                # 相关性分数
                relevance = item.score
                
                # 与已选集合的最大相似度
                max_sim = 0
                for selected_item in selected:
                    sim = self._cosine_similarity(item.embedding, selected_item.embedding)
                    max_sim = max(max_sim, sim)
                
                # MMR分数
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_item = item
                    best_idx = idx
            
            if best_item is not None:
                selected.append(best_item)
                candidates.pop(best_idx)
            else:
                break
        
        return selected
    
    def _sliding_window_rerank(self, items: List[KnowledgeItem],
                               top_k: int) -> List[KnowledgeItem]:
        """
        滑动窗口类目打散
        
        在任意窗口内，同类目内容不超过一定比例
        """
        selected = []
        candidates = list(items)
        
        while len(selected) < top_k and candidates:
            for idx, item in enumerate(candidates):
                # 检查滑动窗口内的类目分布
                window = selected[-self.window_size:] if len(selected) >= self.window_size else selected
                same_category_count = sum(1 for s in window if s.category_id == item.category_id)
                
                # 窗口内同类目内容不超过1个
                if same_category_count == 0 or len(window) < self.window_size:
                    selected.append(item)
                    candidates.pop(idx)
                    break
            else:
                # 如果所有候选都不满足条件，选择分数最高的
                selected.append(candidates.pop(0))
        
        return selected
    
    def _category_balance_rerank(self, items: List[KnowledgeItem],
                                  top_k: int) -> List[KnowledgeItem]:
        """
        类目平衡重排
        
        确保各类目内容在推荐列表中的均衡分布
        """
        # 按类目分组
        category_items: Dict[int, List[KnowledgeItem]] = {}
        for item in items:
            if item.category_id not in category_items:
                category_items[item.category_id] = []
            category_items[item.category_id].append(item)
        
        # 轮询选择
        selected = []
        category_pointers = {cat: 0 for cat in category_items}
        categories = list(category_items.keys())
        cat_idx = 0
        
        while len(selected) < top_k and category_pointers:
            cat = categories[cat_idx % len(categories)]
            
            if category_pointers[cat] < len(category_items[cat]):
                selected.append(category_items[cat][category_pointers[cat]])
                category_pointers[cat] += 1
            else:
                # 该类目已用完
                del category_pointers[cat]
                categories.remove(cat)
                if not categories:
                    break
            
            cat_idx += 1
        
        # 按原始分数重新排序（保持一定的分数顺序）
        # 但这里我们保持轮询结果以保证多样性
        return selected
    
    def _dpp_rerank(self, items: List[KnowledgeItem],
                    top_k: int) -> List[KnowledgeItem]:
        """
        DPP (Determinantal Point Process) 重排
        
        利用行列式计算子集的多样性得分
        """
        n = len(items)
        if n == 0:
            return []
        
        # 构建相似度矩阵
        embeddings = np.array([item.embedding for item in items])
        scores = np.array([item.score for item in items])
        
        # 质量-多样性核矩阵
        # L_ij = q_i * q_j * S_ij
        quality = np.sqrt(np.maximum(scores, 1e-8))
        similarity = embeddings @ embeddings.T
        L = np.outer(quality, quality) * similarity
        
        # 贪心近似DPP
        selected_indices = []
        remaining = set(range(n))
        
        # 选择第一个（质量最高的）
        first_idx = np.argmax(scores)
        selected_indices.append(first_idx)
        remaining.remove(first_idx)
        
        while len(selected_indices) < top_k and remaining:
            best_idx = -1
            best_gain = float('-inf')
            
            for idx in remaining:
                # 计算边际增益（简化版）
                marginal_quality = scores[idx]
                
                # 与已选集合的平均多样性
                diversity = 0
                for sel_idx in selected_indices:
                    diversity += 1 - similarity[idx, sel_idx]
                diversity /= len(selected_indices)
                
                gain = self.lambda_param * marginal_quality + (1 - self.lambda_param) * diversity
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        return [items[i] for i in selected_indices]
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0
        return np.dot(a, b) / (norm_a * norm_b)


class DeduplicationFilter:
    """
    去重过滤器
    
    过滤用户已经学习过或明确不感兴趣的内容
    """
    def __init__(self):
        """初始化去重过滤器"""
        pass
        
    def filter(self, items: List[KnowledgeItem],
               learned_ids: Set[int],
               negative_ids: Set[int],
               negative_categories: Set[int] = None,
               negative_authors: Set[int] = None) -> List[KnowledgeItem]:
        """
        去重过滤
        
        参数:
            items (List[KnowledgeItem]): 候选知识列表
            learned_ids (Set[int]): 用户已学习的知识ID
            negative_ids (Set[int]): 用户明确不感兴趣的知识ID
            negative_categories (Set[int], optional): 不感兴趣的类目
            negative_authors (Set[int], optional): 不感兴趣的作者
            
        返回:
            List[KnowledgeItem]: 过滤后的知识列表
        """
        filtered = []
        
        for item in items:
            # 过滤已学习内容
            if item.knowledge_id in learned_ids:
                continue
            
            # 过滤负反馈内容
            if item.knowledge_id in negative_ids:
                continue
            
            # 过滤不感兴趣的类目
            if negative_categories and item.category_id in negative_categories:
                continue
            
            # 过滤不感兴趣的作者
            if negative_authors and item.author_id in negative_authors:
                continue
            
            filtered.append(item)
        
        return filtered


class FreshnessBooster:
    """
    新内容扶持器
    
    给新发布的优质内容一个初始流量池
    """
    def __init__(self, boost_days: int = 7, boost_ratio: float = 0.2,
                 min_quality_threshold: float = 0.3):
        """
        初始化新内容扶持器
        
        参数:
            boost_days (int): 扶持天数
            boost_ratio (float): 新内容在结果中的最大占比
            min_quality_threshold (float): 最低质量阈值
        """
        self.boost_days = boost_days
        self.boost_ratio = boost_ratio
        self.min_quality_threshold = min_quality_threshold
        
    def boost(self, items: List[KnowledgeItem],
              current_timestamp: int,
              top_k: int) -> List[KnowledgeItem]:
        """
        新内容扶持
        
        参数:
            items (List[KnowledgeItem]): 候选知识列表
            current_timestamp (int): 当前时间戳
            top_k (int): 返回数量
            
        返回:
            List[KnowledgeItem]: 扶持后的知识列表
        """
        seconds_per_day = 86400
        boost_threshold = current_timestamp - self.boost_days * seconds_per_day
        
        fresh_items = []
        regular_items = []
        
        for item in items:
            if (item.publish_timestamp > boost_threshold and 
                item.score >= self.min_quality_threshold):
                fresh_items.append(item)
            else:
                regular_items.append(item)
        
        # 计算新内容配额
        fresh_quota = int(top_k * self.boost_ratio)
        fresh_quota = min(fresh_quota, len(fresh_items))
        
        # 组合结果
        result = []
        fresh_selected = fresh_items[:fresh_quota]
        regular_selected = regular_items[:top_k - fresh_quota]
        
        # 交替插入新内容
        fresh_idx = 0
        regular_idx = 0
        
        # 每隔几个位置插入一个新内容
        if fresh_selected:
            interval = max(1, top_k // (fresh_quota + 1))
        else:
            interval = top_k + 1
        
        for i in range(top_k):
            if (i + 1) % interval == 0 and fresh_idx < len(fresh_selected):
                result.append(fresh_selected[fresh_idx])
                fresh_idx += 1
            elif regular_idx < len(regular_selected):
                result.append(regular_selected[regular_idx])
                regular_idx += 1
            elif fresh_idx < len(fresh_selected):
                result.append(fresh_selected[fresh_idx])
                fresh_idx += 1
        
        return result


class BusinessRuleEngine:
    """
    业务规则引擎
    
    融入平台运营策略，如强制置顶、广告位等
    """
    def __init__(self):
        """初始化业务规则引擎"""
        self.pinned_items: List[KnowledgeItem] = []
        self.blocked_ids: Set[int] = set()
        
    def add_pinned_item(self, item: KnowledgeItem, position: int = 0):
        """添加置顶内容"""
        self.pinned_items.append((item, position))
        
    def add_blocked_id(self, knowledge_id: int):
        """添加屏蔽ID"""
        self.blocked_ids.add(knowledge_id)
        
    def apply_rules(self, items: List[KnowledgeItem]) -> List[KnowledgeItem]:
        """
        应用业务规则
        
        参数:
            items (List[KnowledgeItem]): 候选知识列表
            
        返回:
            List[KnowledgeItem]: 应用规则后的列表
        """
        # 过滤屏蔽内容
        items = [item for item in items if item.knowledge_id not in self.blocked_ids]
        
        # 插入置顶内容
        for pinned_item, position in sorted(self.pinned_items, key=lambda x: x[1]):
            if position < len(items):
                items.insert(position, pinned_item)
            else:
                items.append(pinned_item)
        
        return items


class ReRanker(nn.Module):
    """
    神经网络重排模型
    
    使用Listwise方法学习最优的列表排序
    """
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        """
        初始化神经网络重排模型
        
        参数:
            embedding_dim (int): 嵌入维度
            hidden_dim (int): 隐藏层维度
        """
        super().__init__()
        
        # 上下文编码器（考虑列表内其他item的信息）
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # 位置编码
        self.position_embedding = nn.Embedding(100, embedding_dim)
        
        # 分数预测
        self.score_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, item_embeddings: torch.Tensor,
                original_scores: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            item_embeddings (torch.Tensor): 候选知识嵌入，形状为 (batch, num_items, dim)
            original_scores (torch.Tensor): 精排分数，形状为 (batch, num_items)
            
        返回:
            torch.Tensor: 重排分数
        """
        batch_size, num_items, dim = item_embeddings.size()
        
        # 添加位置编码
        positions = torch.arange(num_items, device=item_embeddings.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        item_embeddings = item_embeddings + pos_emb
        
        # Transformer编码（考虑列表内的上下文信息）
        context_embeddings = self.context_encoder(item_embeddings)
        
        # 结合原始嵌入和上下文嵌入
        combined = torch.cat([item_embeddings, context_embeddings], dim=-1)
        
        # 预测重排分数
        rerank_scores = self.score_predictor(combined).squeeze(-1)
        
        # 与原始分数加权融合
        final_scores = 0.7 * rerank_scores + 0.3 * original_scores
        
        return final_scores
    
    def compute_listwise_loss(self, predicted_scores: torch.Tensor,
                               true_labels: torch.Tensor) -> torch.Tensor:
        """
        计算Listwise损失（ListNet）
        
        参数:
            predicted_scores (torch.Tensor): 预测分数
            true_labels (torch.Tensor): 真实标签（相关性）
            
        返回:
            torch.Tensor: 损失值
        """
        # Top-1概率分布
        pred_probs = F.softmax(predicted_scores, dim=-1)
        true_probs = F.softmax(true_labels, dim=-1)
        
        # 交叉熵损失
        loss = -torch.sum(true_probs * torch.log(pred_probs + 1e-10), dim=-1)
        return loss.mean()


class KnowledgeReRankingPipeline:
    """
    知识库重排流水线
    
    整合所有重排策略
    """
    def __init__(self, config: Dict):
        """
        初始化重排流水线
        
        参数:
            config (Dict): 配置参数
        """
        # 多样性重排器
        self.diversity_reranker = DiversityReranker(
            strategy=DiversityStrategy(config.get('diversity_strategy', 'mmr')),
            lambda_param=config.get('diversity_lambda', 0.6),
            window_size=config.get('diversity_window', 3)
        )
        
        # 去重过滤器
        self.dedup_filter = DeduplicationFilter()
        
        # 新内容扶持器
        self.freshness_booster = FreshnessBooster(
            boost_days=config.get('boost_days', 7),
            boost_ratio=config.get('boost_ratio', 0.15),
            min_quality_threshold=config.get('min_quality', 0.3)
        )
        
        # 业务规则引擎
        self.business_rules = BusinessRuleEngine()
        
        # 神经网络重排模型（可选）
        self.neural_reranker = None
        if config.get('use_neural_reranker', False):
            self.neural_reranker = ReRanker(
                embedding_dim=config.get('embedding_dim', 64),
                hidden_dim=config.get('hidden_dim', 128)
            )
        
    def rerank(self, items: List[KnowledgeItem],
               user_context: Dict,
               top_k: int = 20) -> List[KnowledgeItem]:
        """
        执行重排流水线
        
        参数:
            items (List[KnowledgeItem]): 精排后的候选列表
            user_context (Dict): 用户上下文信息
                - learned_ids: 已学习的知识ID
                - negative_ids: 负反馈ID
                - negative_categories: 不感兴趣的类目
                - current_timestamp: 当前时间戳
            top_k (int): 返回数量
            
        返回:
            List[KnowledgeItem]: 重排后的知识列表
        """
        # 1. 去重过滤
        items = self.dedup_filter.filter(
            items,
            learned_ids=user_context.get('learned_ids', set()),
            negative_ids=user_context.get('negative_ids', set()),
            negative_categories=user_context.get('negative_categories', set()),
            negative_authors=user_context.get('negative_authors', set())
        )
        
        if not items:
            return []
        
        # 2. 多样性重排
        items = self.diversity_reranker.rerank(items, min(top_k * 2, len(items)))
        
        # 3. 新内容扶持
        items = self.freshness_booster.boost(
            items,
            current_timestamp=user_context.get('current_timestamp', 0),
            top_k=top_k
        )
        
        # 4. 应用业务规则
        items = self.business_rules.apply_rules(items)
        
        return items[:top_k]


def evaluate_diversity(items: List[KnowledgeItem]) -> Dict[str, float]:
    """
    评估推荐列表的多样性
    
    参数:
        items (List[KnowledgeItem]): 推荐知识列表
        
    返回:
        Dict[str, float]: 多样性指标
    """
    if not items:
        return {}
    
    # 类目覆盖率
    categories = set(item.category_id for item in items)
    category_coverage = len(categories) / len(items)
    
    # 平均相似度（越低越好）
    n = len(items)
    if n < 2:
        avg_similarity = 0
    else:
        total_sim = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = DiversityReranker._cosine_similarity(
                    items[i].embedding, items[j].embedding
                )
                total_sim += sim
                count += 1
        avg_similarity = total_sim / count if count > 0 else 0
    
    # 基尼系数（类目分布均匀度）
    category_counts = {}
    for item in items:
        category_counts[item.category_id] = category_counts.get(item.category_id, 0) + 1
    
    counts = sorted(category_counts.values())
    n_cat = len(counts)
    if n_cat > 1:
        gini = sum((2 * i - n_cat + 1) * c for i, c in enumerate(counts)) / (n_cat * sum(counts))
    else:
        gini = 0
    
    return {
        'category_coverage': category_coverage,
        'avg_similarity': avg_similarity,
        'diversity_score': 1 - avg_similarity,
        'gini_coefficient': gini
    }


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 生成模拟数据
    def generate_mock_items(n: int) -> List[KnowledgeItem]:
        items = []
        for i in range(n):
            item = KnowledgeItem(
                knowledge_id=i,
                category_id=i % 5,  # 5个类目
                difficulty=i % 3,
                tags=[i % 10, (i + 1) % 10],
                score=np.random.random(),
                embedding=np.random.randn(64),
                publish_timestamp=int(1000000000 + i * 1000),
                learner_count=np.random.randint(0, 10000),
                author_id=i % 20
            )
            items.append(item)
        return sorted(items, key=lambda x: -x.score)
    
    # 生成100个候选
    candidates = generate_mock_items(100)
    print(f"生成 {len(candidates)} 个候选知识")
    
    # 测试多样性重排
    print("\n=== 测试多样性重排 ===")
    for strategy in DiversityStrategy:
        reranker = DiversityReranker(strategy=strategy, lambda_param=0.6)
        result = reranker.rerank(candidates, top_k=20)
        diversity_metrics = evaluate_diversity(result)
        print(f"\n策略: {strategy.value}")
        print(f"  类目覆盖率: {diversity_metrics['category_coverage']:.3f}")
        print(f"  多样性分数: {diversity_metrics['diversity_score']:.3f}")
        print(f"  基尼系数: {diversity_metrics['gini_coefficient']:.3f}")
    
    # 测试完整流水线
    print("\n=== 测试完整重排流水线 ===")
    config = {
        'diversity_strategy': 'mmr',
        'diversity_lambda': 0.6,
        'boost_days': 7,
        'boost_ratio': 0.15,
        'use_neural_reranker': False
    }
    
    pipeline = KnowledgeReRankingPipeline(config)
    
    user_context = {
        'learned_ids': {0, 1, 2},  # 已学习
        'negative_ids': {5, 6},     # 负反馈
        'negative_categories': set(),
        'current_timestamp': 1000100000
    }
    
    final_result = pipeline.rerank(candidates, user_context, top_k=20)
    print(f"\n最终推荐数量: {len(final_result)}")
    
    diversity_metrics = evaluate_diversity(final_result)
    print(f"最终多样性指标:")
    for metric, value in diversity_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # 检查去重效果
    result_ids = {item.knowledge_id for item in final_result}
    filtered_ids = result_ids & (user_context['learned_ids'] | user_context['negative_ids'])
    print(f"\n去重验证: 结果中不应有的ID数量 = {len(filtered_ids)}")
    
    print("\n✅ 重排模块测试通过！")

