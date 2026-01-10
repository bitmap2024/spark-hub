# 知识库推荐系统 (Knowledge Recommendation System)

基于深度学习的知识库推荐系统，采用工业级三层架构设计，参考小红书等平台的推荐系统最佳实践。

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        推荐请求 (Request)                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     📥 召回层 (Retrieval)                        │
│                                                                 │
│   双塔模型 (Two-Tower)                                          │
│   ├── 用户塔: 用户特征 + 上下文 + 历史行为                        │
│   └── 知识塔: 知识特征 + 文本语义                                 │
│                                                                 │
│   向量检索: FAISS / Milvus                                       │
│   范围: 全库 50000+ → 500 候选                                   │
│   延迟: < 10ms                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     🎯 精排层 (Ranking)                          │
│                                                                 │
│   PLE模型 (Progressive Layered Extraction)                      │
│   ├── 共享专家层: 学习通用特征                                    │
│   ├── 任务专属专家: 解决任务冲突                                  │
│   └── 多任务预测:                                                │
│       • 点击率 (pCTR)                                           │
│       • 收藏率 (pCollect)                                       │
│       • 完成率 (pComplete)                                      │
│       • 分享率 (pShare)                                         │
│       • 停留时长 (pDuration)                                    │
│                                                                 │
│   特征交叉: DCN (Deep & Cross Network)                          │
│   用户兴趣: Target Attention (DIN)                              │
│   范围: 500 候选 → 100 候选                                     │
│   延迟: < 50ms                                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     🔄 重排层 (Re-ranking)                       │
│                                                                 │
│   多样性打散 (Diversity)                                         │
│   ├── MMR: Maximal Marginal Relevance                          │
│   ├── DPP: Determinantal Point Process                         │
│   └── 滑动窗口类目打散                                           │
│                                                                 │
│   业务策略:                                                      │
│   ├── 去重过滤 (已学习/负反馈)                                   │
│   ├── 新内容扶持 (初始流量池)                                    │
│   └── 业务规则 (置顶/屏蔽)                                       │
│                                                                 │
│   范围: 100 候选 → 20 展示                                      │
│   延迟: < 5ms                                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        推荐结果 (Response)                       │
│                                                                 │
│   • 推荐知识ID列表                                               │
│   • 预测分数                                                     │
│   • 推荐解释                                                     │
│   • 多样性指标                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
knowedge_recomment/
├── feature_encoder.py      # 特征编码模块
│   ├── NumericalBucketing  # 数值分桶编码
│   ├── CategoricalEncoder  # 类别特征编码
│   ├── MultiHotEncoder     # 多标签编码
│   ├── TextEncoder         # 文本特征编码
│   ├── UserFeatureEncoder  # 用户特征编码器
│   ├── KnowledgeFeatureEncoder  # 知识特征编码器
│   ├── ContextFeatureEncoder    # 上下文特征编码器
│   └── SequenceEncoder     # 序列特征编码器 (Target Attention)
│
├── retrieval_model.py      # 召回层模型
│   ├── UserTower           # 用户塔
│   ├── KnowledgeTower      # 知识塔
│   ├── TwoTowerRetrieval   # 双塔召回模型
│   ├── HardNegativeMiner   # 困难负样本挖掘
│   └── MultiChannelRetrieval  # 多路召回
│
├── ranking_model.py        # 精排层模型
│   ├── Expert              # 专家网络
│   ├── GatingNetwork       # 门控网络
│   ├── ExtractionNetwork   # PLE提取网络
│   ├── PLERankingModel     # PLE精排模型
│   └── TargetAttentionDIN  # DIN注意力模块
│
├── reranking_model.py      # 重排层
│   ├── DiversityReranker   # 多样性重排器
│   ├── DeduplicationFilter # 去重过滤器
│   ├── FreshnessBooster    # 新内容扶持器
│   ├── BusinessRuleEngine  # 业务规则引擎
│   ├── ReRanker            # 神经网络重排模型
│   └── KnowledgeReRankingPipeline  # 重排流水线
│
├── recommendation_service.py  # 推荐服务
│   ├── KnowledgeRecommendationService  # 主服务类
│   ├── VectorIndex         # 向量索引
│   └── RecommendationEvaluator  # 效果评估器
│
├── requirements.txt        # 依赖配置
└── README.md              # 项目文档
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化服务

```python
from recommendation_service import KnowledgeRecommendationService, RecommendationRequest, create_default_config

# 创建配置
config = create_default_config()

# 初始化服务
service = KnowledgeRecommendationService(config)

# 构建知识库索引
knowledge_data = [
    {
        'knowledge_id': 1,
        'category': 5,
        'difficulty': 2,
        'tags': [10, 20, 30],
        'title_tokens': [100, 200, 300],
        'learner_count': 5000,
        'rating': 4.5,
        'duration': 60,
        'days_since_publish': 30
    },
    # ... 更多知识
]
service.build_knowledge_index(knowledge_data)
```

### 3. 执行推荐

```python
# 创建推荐请求
request = RecommendationRequest(
    user_id=12345,
    user_features={
        'level': 2,              # 用户等级
        'profession': 5,         # 职业
        'interest_tags': [1, 2, 3],  # 兴趣标签
        'learning_days': 100,    # 学习天数
        'completion_rate': 75    # 完成率
    },
    context_features={
        'hour': 14,              # 当前小时
        'weekday': 3,            # 星期几
        'device': 1,             # 设备类型
        'platform': 0            # 平台
    },
    history_ids=[10, 20, 30, 40, 50],  # 历史学习ID
    num_results=20,             # 返回数量
    exclude_ids=[100, 101]      # 排除ID
)

# 获取推荐结果
result = service.recommend(request)

print(f"推荐知识: {result.knowledge_ids}")
print(f"推荐分数: {result.scores}")
print(f"推荐解释: {result.explanations}")
print(f"响应延迟: {result.latency_ms:.2f}ms")
```

## 📊 特征工程

### 用户特征

| 特征名 | 类型 | 说明 | 编码方式 |
|--------|------|------|----------|
| user_id | 类别 | 用户ID | Embedding |
| level | 类别 | 学习等级 | Embedding |
| profession | 类别 | 职业领域 | Embedding |
| interest_tags | 多标签 | 兴趣标签 | Multi-hot + Mean Pooling |
| learning_days | 数值 | 学习天数 | 对数分桶 + Embedding |
| completion_rate | 数值 | 完成率 | 线性分桶 + Embedding |

### 知识特征

| 特征名 | 类型 | 说明 | 编码方式 |
|--------|------|------|----------|
| knowledge_id | 类别 | 知识ID | Embedding |
| category | 类别 | 知识类别 | Embedding |
| difficulty | 类别 | 难度级别 | Embedding |
| tags | 多标签 | 内容标签 | Multi-hot + Mean Pooling |
| title | 文本 | 标题 | BiLSTM |
| learner_count | 数值 | 学习人数 | 对数分桶 |
| rating | 数值 | 平均评分 | 线性分桶 |
| duration | 数值 | 学习时长 | 对数分桶 |
| freshness | 数值 | 新鲜度 | 对数分桶 |

### 上下文特征

| 特征名 | 类型 | 说明 |
|--------|------|------|
| hour | 类别 | 请求小时 |
| weekday | 类别 | 星期几 |
| device | 类别 | 设备类型 |
| platform | 类别 | 访问平台 |

## 📈 评估指标

### 离线指标

- **AUC**: 排序质量
- **LogLoss**: 预测准确性
- **NDCG@K**: 位置敏感排序质量
- **Precision@K / Recall@K**: 准确率/召回率
- **Hit Rate@K**: 命中率

### 在线指标

- **CTR**: 点击率
- **收藏率 / 完成率 / 分享率**: 交互指标
- **Dwell Time**: 停留时长
- **Retention**: 留存率

### 生态指标

- **Diversity Score**: 多样性分数
- **Coverage**: 内容覆盖率
- **Gini Coefficient**: 流量公平性

## ⚙️ 配置参数

```python
config = {
    # 模型维度
    'embedding_dim': 64,        # 嵌入维度
    'expert_dim': 128,          # 专家网络维度
    
    # PLE配置
    'num_extraction_layers': 2, # 提取层数
    'num_shared_experts': 3,    # 共享专家数
    'num_task_experts': 2,      # 任务专属专家数
    
    # 召回配置
    'recall_num': 500,          # 召回数量
    'max_history_len': 50,      # 最大历史长度
    
    # 重排配置
    'reranking_config': {
        'diversity_strategy': 'mmr',  # 多样性策略
        'diversity_lambda': 0.6,      # 多样性权重
        'boost_days': 7,              # 新内容扶持天数
        'boost_ratio': 0.15           # 新内容占比
    }
}
```

## 🔧 模型训练

### 召回模型训练

```python
from retrieval_model import TwoTowerRetrieval, train_two_tower

model = TwoTowerRetrieval(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
history = train_two_tower(model, train_loader, optimizer, num_epochs=10)

# 保存
torch.save(model.state_dict(), 'retrieval_model.pt')
```

### 精排模型训练

```python
from ranking_model import PLERankingModel, train_ple_model

model = PLERankingModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练
history = train_ple_model(model, train_loader, optimizer, num_epochs=10)

# 保存
torch.save(model.state_dict(), 'ranking_model.pt')
```

## 📚 参考文献

1. **双塔模型**: Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (Google, 2019)
2. **PLE**: Progressive Layered Extraction: A Novel Multi-Task Learning Model for Personalized Recommendations (Tencent, 2020)
3. **DIN**: Deep Interest Network for Click-Through Rate Prediction (Alibaba, 2018)
4. **DCN**: Deep & Cross Network for Ad Click Predictions (Google, 2017)
5. **MMoE**: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (Google, 2018)

## 📝 License

MIT License
