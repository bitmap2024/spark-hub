"""
知识库推荐系统 - 推荐服务

本模块整合了推荐系统的所有组件，提供完整的推荐服务：
- 召回层：双塔模型快速召回候选
- 精排层：PLE多任务学习精细排序
- 重排层：多样性、去重、新内容扶持

支持：
- 在线推荐服务
- 离线评估
- 模型训练和更新
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import logging
from dataclasses import dataclass
from enum import Enum
import json

# 内部模块导入
from feature_encoder import UserFeatureEncoder, KnowledgeFeatureEncoder, ContextFeatureEncoder
from retrieval_model import TwoTowerRetrieval
from ranking_model import PLERankingModel
from reranking_model import (
    KnowledgeReRankingPipeline, KnowledgeItem, 
    evaluate_diversity, DiversityStrategy
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """推荐类型枚举"""
    PERSONALIZED = "personalized"      # 个性化推荐（首页）
    SIMILAR = "similar"                 # 相似推荐
    TRENDING = "trending"               # 热门趋势
    NEW_ARRIVAL = "new_arrival"         # 新上架
    CONTINUE_LEARNING = "continue"      # 继续学习


@dataclass
class RecommendationRequest:
    """推荐请求数据类"""
    user_id: int
    user_features: Dict
    context_features: Dict
    history_ids: List[int]
    recommendation_type: RecommendationType = RecommendationType.PERSONALIZED
    num_results: int = 20
    exclude_ids: List[int] = None
    filter_categories: List[int] = None
    
    def __post_init__(self):
        if self.exclude_ids is None:
            self.exclude_ids = []
        if self.filter_categories is None:
            self.filter_categories = []


@dataclass
class RecommendationResult:
    """推荐结果数据类"""
    knowledge_ids: List[int]
    scores: List[float]
    explanations: List[str]
    diversity_metrics: Dict[str, float]
    latency_ms: float
    
    def to_dict(self) -> Dict:
        return {
            'knowledge_ids': self.knowledge_ids,
            'scores': self.scores,
            'explanations': self.explanations,
            'diversity_metrics': self.diversity_metrics,
            'latency_ms': self.latency_ms
        }


class VectorIndex:
    """
    向量索引（模拟FAISS/Milvus）
    
    用于快速检索最相似的知识向量
    """
    def __init__(self, embedding_dim: int):
        """
        初始化向量索引
        
        参数:
            embedding_dim (int): 嵌入维度
        """
        self.embedding_dim = embedding_dim
        self.vectors = None
        self.ids = None
        
    def build_index(self, vectors: np.ndarray, ids: np.ndarray):
        """
        构建索引
        
        参数:
            vectors (np.ndarray): 知识向量，形状为 (num_items, dim)
            ids (np.ndarray): 知识ID
        """
        # 归一化向量（用于余弦相似度）
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / (norms + 1e-8)
        self.ids = ids
        logger.info(f"向量索引构建完成，共 {len(ids)} 条记录")
        
    def search(self, query_vector: np.ndarray, top_k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        搜索最相似的向量
        
        参数:
            query_vector (np.ndarray): 查询向量，形状为 (dim,) 或 (batch, dim)
            top_k (int): 返回数量
            
        返回:
            Tuple: (知识ID, 相似度分数)
        """
        if self.vectors is None:
            return np.array([]), np.array([])
        
        # 归一化查询向量
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / (norms + 1e-8)
        
        # 计算余弦相似度
        similarities = np.dot(query_vector, self.vectors.T)
        
        # 获取top-k
        top_k = min(top_k, len(self.ids))
        top_indices = np.argsort(-similarities, axis=1)[:, :top_k]
        
        if query_vector.shape[0] == 1:
            return self.ids[top_indices[0]], similarities[0, top_indices[0]]
        
        return self.ids[top_indices], similarities[np.arange(len(similarities))[:, None], top_indices]


class KnowledgeRecommendationService:
    """
    知识库推荐服务
    
    提供端到端的推荐能力
    """
    def __init__(self, config: Dict):
        """
        初始化推荐服务
        
        参数:
            config (Dict): 配置参数
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 初始化模型
        self._init_models(config)
        
        # 初始化向量索引
        self.vector_index = VectorIndex(config['embedding_dim'])
        
        # 初始化重排流水线
        self.reranking_pipeline = KnowledgeReRankingPipeline(config.get('reranking_config', {}))
        
        # 知识库元数据（实际应用中从数据库加载）
        self.knowledge_metadata: Dict[int, Dict] = {}
        
        # 统计信息
        self.request_count = 0
        self.total_latency = 0
        
        logger.info(f"推荐服务初始化完成，设备: {self.device}")
        
    def _init_models(self, config: Dict):
        """初始化推荐模型"""
        # 召回模型
        self.retrieval_model = TwoTowerRetrieval(config)
        self.retrieval_model.to(self.device)
        self.retrieval_model.eval()
        
        # 精排模型
        self.ranking_model = PLERankingModel(config)
        self.ranking_model.to(self.device)
        self.ranking_model.eval()
        
    def load_models(self, retrieval_path: str, ranking_path: str):
        """
        加载模型权重
        
        参数:
            retrieval_path (str): 召回模型路径
            ranking_path (str): 精排模型路径
        """
        self.retrieval_model.load_state_dict(torch.load(retrieval_path, map_location=self.device))
        self.ranking_model.load_state_dict(torch.load(ranking_path, map_location=self.device))
        logger.info("模型加载完成")
        
    def build_knowledge_index(self, knowledge_data: List[Dict]):
        """
        构建知识库索引
        
        参数:
            knowledge_data (List[Dict]): 知识数据列表
        """
        logger.info(f"开始构建知识库索引，共 {len(knowledge_data)} 条知识")
        
        # 提取特征并计算向量
        vectors = []
        ids = []
        
        batch_size = self.config.get('index_batch_size', 128)
        
        for i in range(0, len(knowledge_data), batch_size):
            batch = knowledge_data[i:i+batch_size]
            
            # 准备特征
            knowledge_features = self._prepare_knowledge_features(batch)
            
            # 计算向量
            with torch.no_grad():
                knowledge_vector = self.retrieval_model.get_knowledge_embedding(knowledge_features)
                vectors.append(knowledge_vector.cpu().numpy())
            
            # 记录ID和元数据
            for item in batch:
                ids.append(item['knowledge_id'])
                self.knowledge_metadata[item['knowledge_id']] = item
        
        # 构建索引
        vectors = np.vstack(vectors)
        ids = np.array(ids)
        self.vector_index.build_index(vectors, ids)
        
        logger.info(f"知识库索引构建完成")
        
    def _prepare_knowledge_features(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """准备知识特征张量"""
        features = {
            'knowledge_id': torch.tensor([item['knowledge_id'] for item in batch]).to(self.device),
            'category': torch.tensor([item.get('category', 0) for item in batch]).to(self.device),
            'difficulty': torch.tensor([item.get('difficulty', 0) for item in batch]).to(self.device),
            'tags': torch.tensor([item.get('tags', [0]*5)[:5] + [0]*(5-len(item.get('tags', []))) for item in batch]).to(self.device),
            'title_tokens': torch.tensor([item.get('title_tokens', [0]*20)[:20] + [0]*(20-len(item.get('title_tokens', []))) for item in batch]).to(self.device),
            'title_lengths': torch.tensor([min(len(item.get('title_tokens', [1])), 20) for item in batch]).to(self.device),
            'learner_count': torch.tensor([item.get('learner_count', 0) for item in batch]).to(self.device),
            'rating': torch.tensor([item.get('rating', 0.0) for item in batch], dtype=torch.float).to(self.device),
            'duration': torch.tensor([item.get('duration', 0) for item in batch]).to(self.device),
            'days_since_publish': torch.tensor([item.get('days_since_publish', 0) for item in batch]).to(self.device)
        }
        return features
        
    def _prepare_user_features(self, request: RecommendationRequest) -> Dict[str, torch.Tensor]:
        """准备用户特征张量"""
        features = {
            'user_id': torch.tensor([request.user_id]).to(self.device),
            'level': torch.tensor([request.user_features.get('level', 0)]).to(self.device),
            'profession': torch.tensor([request.user_features.get('profession', 0)]).to(self.device),
            'interest_tags': torch.tensor([request.user_features.get('interest_tags', [0]*10)]).to(self.device),
            'learning_days': torch.tensor([request.user_features.get('learning_days', 0)]).to(self.device),
            'completion_rate': torch.tensor([request.user_features.get('completion_rate', 0)]).to(self.device)
        }
        return features
        
    def _prepare_context_features(self, request: RecommendationRequest) -> Dict[str, torch.Tensor]:
        """准备上下文特征张量"""
        features = {
            'hour': torch.tensor([request.context_features.get('hour', 12)]).to(self.device),
            'weekday': torch.tensor([request.context_features.get('weekday', 0)]).to(self.device),
            'device': torch.tensor([request.context_features.get('device', 0)]).to(self.device),
            'platform': torch.tensor([request.context_features.get('platform', 0)]).to(self.device)
        }
        return features

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        """
        执行推荐
        
        参数:
            request (RecommendationRequest): 推荐请求
            
        返回:
            RecommendationResult: 推荐结果
        """
        start_time = time.time()
        
        try:
            # 1. 召回阶段
            recall_ids, recall_scores = self._recall(request)
            logger.debug(f"召回完成，返回 {len(recall_ids)} 个候选")
            
            # 2. 精排阶段
            ranking_results = self._rank(request, recall_ids)
            logger.debug(f"精排完成")
            
            # 3. 重排阶段
            final_items = self._rerank(request, ranking_results)
            logger.debug(f"重排完成，最终返回 {len(final_items)} 个结果")
            
            # 4. 生成解释
            explanations = self._generate_explanations(final_items)
            
            # 5. 计算多样性指标
            diversity_metrics = evaluate_diversity(final_items) if final_items else {}
            
            latency_ms = (time.time() - start_time) * 1000
            
            # 更新统计
            self.request_count += 1
            self.total_latency += latency_ms
            
            return RecommendationResult(
                knowledge_ids=[item.knowledge_id for item in final_items],
                scores=[item.score for item in final_items],
                explanations=explanations,
                diversity_metrics=diversity_metrics,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(f"推荐服务异常: {str(e)}")
            raise
            
    def _recall(self, request: RecommendationRequest) -> Tuple[np.ndarray, np.ndarray]:
        """召回阶段"""
        # 准备特征
        user_features = self._prepare_user_features(request)
        context_features = self._prepare_context_features(request)
        
        # 准备历史序列
        history = request.history_ids[:50] if request.history_ids else [0]
        history = history + [0] * (50 - len(history))
        history_ids = torch.tensor([history]).to(self.device)
        history_lengths = torch.tensor([min(len(request.history_ids), 50)]).to(self.device)
        
        # 计算用户向量
        with torch.no_grad():
            user_vector = self.retrieval_model.get_user_embedding(
                user_features, context_features, history_ids, history_lengths
            )
            user_vector = user_vector.cpu().numpy()[0]
        
        # 向量检索
        recall_num = self.config.get('recall_num', 500)
        recall_ids, recall_scores = self.vector_index.search(user_vector, top_k=recall_num)
        
        # 过滤
        if request.exclude_ids:
            exclude_set = set(request.exclude_ids)
            mask = np.array([id not in exclude_set for id in recall_ids])
            recall_ids = recall_ids[mask]
            recall_scores = recall_scores[mask]
        
        return recall_ids, recall_scores
    
    def _rank(self, request: RecommendationRequest, 
              candidate_ids: np.ndarray) -> List[Tuple[int, Dict[str, float], float]]:
        """精排阶段"""
        if len(candidate_ids) == 0:
            return []
        
        # 准备特征
        user_features = self._prepare_user_features(request)
        context_features = self._prepare_context_features(request)
        
        # 扩展用户和上下文特征以匹配候选数量
        batch_size = len(candidate_ids)
        user_features = {k: v.expand(batch_size, -1) if v.dim() > 1 else v.repeat(batch_size) 
                        for k, v in user_features.items()}
        context_features = {k: v.repeat(batch_size) for k, v in context_features.items()}
        
        # 准备知识特征
        candidate_data = [self.knowledge_metadata.get(int(id), {'knowledge_id': int(id)}) 
                         for id in candidate_ids]
        knowledge_features = self._prepare_knowledge_features(candidate_data)
        
        # 准备历史序列
        history = request.history_ids[:50] if request.history_ids else [0]
        history = history + [0] * (50 - len(history))
        history_ids = torch.tensor([history]).repeat(batch_size, 1).to(self.device)
        history_lengths = torch.tensor([min(len(request.history_ids), 50)]).repeat(batch_size).to(self.device)
        
        # 精排预测
        with torch.no_grad():
            predictions = self.ranking_model(
                user_features, knowledge_features, context_features,
                history_ids, history_lengths
            )
            
            # 计算综合排序分数
            ranking_scores = self.ranking_model.compute_ranking_score(predictions)
            ranking_scores = ranking_scores.cpu().numpy()
            
            # 转换预测结果
            predictions = {k: v.cpu().numpy() for k, v in predictions.items()}
        
        # 组合结果
        results = []
        for i, id in enumerate(candidate_ids):
            task_scores = {k: float(v[i]) for k, v in predictions.items()}
            results.append((int(id), task_scores, float(ranking_scores[i])))
        
        # 按分数排序
        results.sort(key=lambda x: -x[2])
        
        return results
    
    def _rerank(self, request: RecommendationRequest,
                ranking_results: List[Tuple[int, Dict[str, float], float]]) -> List[KnowledgeItem]:
        """重排阶段"""
        # 转换为KnowledgeItem
        items = []
        for knowledge_id, task_scores, score in ranking_results:
            metadata = self.knowledge_metadata.get(knowledge_id, {})
            
            # 获取或生成嵌入向量
            embedding = np.random.randn(self.config['embedding_dim'])
            if hasattr(self, 'knowledge_embeddings') and knowledge_id in self.knowledge_embeddings:
                embedding = self.knowledge_embeddings[knowledge_id]
            
            item = KnowledgeItem(
                knowledge_id=knowledge_id,
                category_id=metadata.get('category', 0),
                difficulty=metadata.get('difficulty', 0),
                tags=metadata.get('tags', []),
                score=score,
                embedding=embedding,
                publish_timestamp=metadata.get('publish_timestamp', 0),
                learner_count=metadata.get('learner_count', 0),
                author_id=metadata.get('author_id', 0)
            )
            items.append(item)
        
        # 用户上下文
        user_context = {
            'learned_ids': set(request.exclude_ids) if request.exclude_ids else set(),
            'negative_ids': set(request.user_features.get('negative_ids', [])),
            'negative_categories': set(request.user_features.get('negative_categories', [])),
            'current_timestamp': request.context_features.get('timestamp', int(time.time()))
        }
        
        # 执行重排
        final_items = self.reranking_pipeline.rerank(items, user_context, request.num_results)
        
        return final_items
    
    def _generate_explanations(self, items: List[KnowledgeItem]) -> List[str]:
        """生成推荐解释"""
        explanations = []
        
        for item in items:
            metadata = self.knowledge_metadata.get(item.knowledge_id, {})
            
            # 根据不同因素生成解释
            reasons = []
            
            if item.score > 0.8:
                reasons.append("与您的学习兴趣高度匹配")
            
            if metadata.get('learner_count', 0) > 1000:
                reasons.append(f"已有{metadata.get('learner_count')}人学习")
            
            if metadata.get('rating', 0) >= 4.5:
                reasons.append("高评分优质内容")
            
            if metadata.get('days_since_publish', 365) < 7:
                reasons.append("最新发布")
            
            if not reasons:
                reasons.append("为您精选")
            
            explanations.append("、".join(reasons))
        
        return explanations
    
    def get_stats(self) -> Dict:
        """获取服务统计信息"""
        return {
            'request_count': self.request_count,
            'avg_latency_ms': self.total_latency / max(1, self.request_count),
            'knowledge_count': len(self.knowledge_metadata),
            'index_size': len(self.vector_index.ids) if self.vector_index.ids is not None else 0
        }


class RecommendationEvaluator:
    """
    推荐效果评估器
    
    提供离线和在线评估指标
    """
    def __init__(self):
        """初始化评估器"""
        pass
    
    def evaluate_offline(self, predictions: List[List[int]], 
                         ground_truth: List[List[int]],
                         k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        离线评估
        
        参数:
            predictions (List[List[int]]): 预测的推荐列表
            ground_truth (List[List[int]]): 真实交互列表
            k_values (List[int]): 评估的K值
            
        返回:
            Dict[str, float]: 评估指标
        """
        metrics = {}
        
        for k in k_values:
            # Precision@K
            precisions = []
            for pred, truth in zip(predictions, ground_truth):
                pred_k = set(pred[:k])
                truth_set = set(truth)
                precision = len(pred_k & truth_set) / k if k > 0 else 0
                precisions.append(precision)
            metrics[f'Precision@{k}'] = np.mean(precisions)
            
            # Recall@K
            recalls = []
            for pred, truth in zip(predictions, ground_truth):
                pred_k = set(pred[:k])
                truth_set = set(truth)
                recall = len(pred_k & truth_set) / len(truth_set) if truth_set else 0
                recalls.append(recall)
            metrics[f'Recall@{k}'] = np.mean(recalls)
            
            # NDCG@K
            ndcgs = []
            for pred, truth in zip(predictions, ground_truth):
                ndcg = self._compute_ndcg(pred[:k], truth)
                ndcgs.append(ndcg)
            metrics[f'NDCG@{k}'] = np.mean(ndcgs)
            
            # Hit Rate@K
            hit_rates = []
            for pred, truth in zip(predictions, ground_truth):
                pred_k = set(pred[:k])
                truth_set = set(truth)
                hit = 1 if pred_k & truth_set else 0
                hit_rates.append(hit)
            metrics[f'HitRate@{k}'] = np.mean(hit_rates)
        
        return metrics
    
    def _compute_ndcg(self, ranked_list: List[int], 
                       ground_truth: List[int]) -> float:
        """计算NDCG"""
        truth_set = set(ground_truth)
        dcg = 0
        for i, item in enumerate(ranked_list):
            if item in truth_set:
                dcg += 1 / np.log2(i + 2)
        
        # 理想DCG
        ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(ranked_list), len(truth_set))))
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0


def create_default_config() -> Dict:
    """创建默认配置"""
    return {
        'embedding_dim': 64,
        'expert_dim': 128,
        'num_extraction_layers': 2,
        'num_shared_experts': 3,
        'num_task_experts': 2,
        'num_knowledge': 50000,
        'max_history_len': 50,
        'recall_num': 500,
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
        },
        'reranking_config': {
            'diversity_strategy': 'mmr',
            'diversity_lambda': 0.6,
            'boost_days': 7,
            'boost_ratio': 0.15,
            'use_neural_reranker': False
        },
        'device': 'cpu'
    }


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("知识库推荐系统测试")
    print("=" * 60)
    
    # 创建配置
    config = create_default_config()
    
    # 创建推荐服务
    print("\n1. 初始化推荐服务...")
    service = KnowledgeRecommendationService(config)
    
    # 模拟知识库数据
    print("\n2. 构建知识库索引...")
    mock_knowledge = []
    for i in range(1000):
        mock_knowledge.append({
            'knowledge_id': i,
            'category': i % 30,
            'difficulty': i % 5,
            'tags': [i % 200, (i + 1) % 200, (i + 2) % 200],
            'title_tokens': [i % 30000 for _ in range(10)],
            'learner_count': np.random.randint(0, 10000),
            'rating': np.random.uniform(3, 5),
            'duration': np.random.randint(10, 120),
            'days_since_publish': np.random.randint(0, 365),
            'author_id': i % 100,
            'publish_timestamp': int(time.time()) - np.random.randint(0, 365 * 24 * 3600)
        })
    
    service.build_knowledge_index(mock_knowledge)
    
    # 创建推荐请求
    print("\n3. 执行推荐请求...")
    request = RecommendationRequest(
        user_id=1,
        user_features={
            'level': 2,
            'profession': 5,
            'interest_tags': [1, 2, 3, 4, 5, 0, 0, 0, 0, 0],
            'learning_days': 100,
            'completion_rate': 75,
            'negative_ids': [500, 501, 502]
        },
        context_features={
            'hour': 14,
            'weekday': 3,
            'device': 1,
            'platform': 0,
            'timestamp': int(time.time())
        },
        history_ids=[10, 20, 30, 40, 50],
        num_results=20,
        exclude_ids=[100, 101, 102]
    )
    
    result = service.recommend(request)
    
    # 打印结果
    print(f"\n推荐结果:")
    print(f"  - 推荐知识数量: {len(result.knowledge_ids)}")
    print(f"  - 响应延迟: {result.latency_ms:.2f} ms")
    print(f"  - 多样性指标: {result.diversity_metrics}")
    
    print(f"\n前5个推荐:")
    for i, (kid, score, explanation) in enumerate(zip(
        result.knowledge_ids[:5], result.scores[:5], result.explanations[:5]
    )):
        print(f"  {i+1}. 知识ID={kid}, 分数={score:.4f}, 解释: {explanation}")
    
    # 获取服务统计
    print(f"\n服务统计:")
    stats = service.get_stats()
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    # 测试评估器
    print("\n4. 测试离线评估...")
    evaluator = RecommendationEvaluator()
    
    # 模拟预测和真实数据
    mock_predictions = [list(range(i, i+20)) for i in range(100)]
    mock_ground_truth = [list(range(i+5, i+15)) for i in range(100)]
    
    metrics = evaluator.evaluate_offline(mock_predictions, mock_ground_truth)
    print(f"\n评估指标:")
    for metric, value in metrics.items():
        print(f"  - {metric}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ 知识库推荐系统测试通过！")
    print("=" * 60)

