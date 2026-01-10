"""
知识库推荐系统 (Knowledge Recommendation System)

一个基于深度学习的工业级知识库推荐系统，采用三层架构：
- 召回层 (Retrieval): 双塔模型快速召回
- 精排层 (Ranking): PLE多任务学习精细排序
- 重排层 (Re-ranking): 多样性打散、去重、新内容扶持

Usage:
    from knowedge_recomment import (
        KnowledgeRecommendationService,
        RecommendationRequest,
        create_default_config
    )
    
    # 初始化服务
    config = create_default_config()
    service = KnowledgeRecommendationService(config)
    
    # 执行推荐
    request = RecommendationRequest(
        user_id=12345,
        user_features={...},
        context_features={...},
        history_ids=[...]
    )
    result = service.recommend(request)
"""

from .recommendation_service import (
    KnowledgeRecommendationService,
    RecommendationRequest,
    RecommendationResult,
    RecommendationType,
    RecommendationEvaluator,
    create_default_config
)

from .retrieval_model import (
    TwoTowerRetrieval,
    UserTower,
    KnowledgeTower,
    HardNegativeMiner,
    MultiChannelRetrieval
)

from .ranking_model import (
    PLERankingModel,
    Expert,
    GatingNetwork,
    ExtractionNetwork,
    TargetAttentionDIN
)

from .reranking_model import (
    KnowledgeReRankingPipeline,
    DiversityReranker,
    DeduplicationFilter,
    FreshnessBooster,
    BusinessRuleEngine,
    KnowledgeItem,
    DiversityStrategy,
    evaluate_diversity
)

from .feature_encoder import (
    UserFeatureEncoder,
    KnowledgeFeatureEncoder,
    ContextFeatureEncoder,
    SequenceEncoder,
    FeatureInteraction,
    NumericalBucketing,
    CategoricalEncoder,
    MultiHotEncoder,
    TextEncoder
)

__version__ = "1.0.0"
__author__ = "Spark Hub Team"

__all__ = [
    # 主服务
    'KnowledgeRecommendationService',
    'RecommendationRequest',
    'RecommendationResult',
    'RecommendationType',
    'RecommendationEvaluator',
    'create_default_config',
    
    # 召回层
    'TwoTowerRetrieval',
    'UserTower',
    'KnowledgeTower',
    'HardNegativeMiner',
    'MultiChannelRetrieval',
    
    # 精排层
    'PLERankingModel',
    'Expert',
    'GatingNetwork',
    'ExtractionNetwork',
    'TargetAttentionDIN',
    
    # 重排层
    'KnowledgeReRankingPipeline',
    'DiversityReranker',
    'DeduplicationFilter',
    'FreshnessBooster',
    'BusinessRuleEngine',
    'KnowledgeItem',
    'DiversityStrategy',
    'evaluate_diversity',
    
    # 特征编码
    'UserFeatureEncoder',
    'KnowledgeFeatureEncoder',
    'ContextFeatureEncoder',
    'SequenceEncoder',
    'FeatureInteraction',
    'NumericalBucketing',
    'CategoricalEncoder',
    'MultiHotEncoder',
    'TextEncoder'
]

