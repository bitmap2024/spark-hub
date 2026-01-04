# Spark Hub 推荐系统算法库

这个目录包含了一系列常用的推荐系统算法实现，基于PyTorch框架构建。这些算法涵盖了从传统的协同过滤到最新的深度学习、图神经网络、多任务学习、强化学习以及大语言模型推荐方法。

## 已实现的算法

### 1. 基础推荐算法

- **协同过滤 (Collaborative Filtering)**
  - 实现文件：`collaborative_filtering.py`
  - 基于用户和物品相似度的传统推荐方法
  - 支持基于用户的协同过滤(UserCF)和基于物品的协同过滤(ItemCF)
  - 使用余弦相似度或皮尔逊相关系数计算相似度

- **基于内容的推荐 (Content-Based Recommendation)**
  - 实现文件：`content_based.py`
  - 基于物品特征相似度的推荐方法
  - 适合处理冷启动问题（新用户或新物品）

- **矩阵分解 (Matrix Factorization)**
  - 实现文件：`matrix_factorization.py`
  - 使用矩阵分解技术建模用户和物品的隐含特征
  - 通过随机梯度下降优化隐向量

- **混合推荐 (Hybrid Recommendation)**
  - 实现文件：`hybrid.py`
  - 结合多种推荐策略的混合推荐系统
  - 通过加权方式融合不同推荐算法的结果

### 2. 深度学习推荐算法

- **神经协同过滤 (Neural Collaborative Filtering, NCF)**
  - 实现文件：`neural_cf.py`
  - 使用神经网络增强协同过滤的表达能力
  - 结合MLP和GMF (Generalized Matrix Factorization)的优点

- **Wide & Deep**
  - 实现文件：`wide_deep.py`
  - Google提出的结合浅层和深层网络的模型，兼顾记忆和泛化能力
  - Wide部分处理特征交叉，Deep部分学习特征的低维表示

- **DeepFM**
  - 实现文件：`deepfm.py`
  - 结合因子分解机(FM)和深度神经网络(DNN)的CTR预测模型
  - 同时建模低阶和高阶特征交互
  - 共享特征嵌入，避免特征工程

- **xDeepFM**
  - 实现文件：`xdeepfm.py`
  - DeepFM的增强版，引入压缩交互网络(CIN)显式建模高阶特征交互
  - 通过外积形式计算特征之间的交互
  - 同时具备显式和隐式高阶特征交互建模能力

- **深度兴趣网络 (Deep Interest Network, DIN)**
  - 实现文件：`din.py`
  - 通过注意力机制对用户的历史行为进行加权建模
  - 能够捕获用户对不同物品的兴趣程度
  - 特别适合电商和广告点击率预测场景

- **深度兴趣演化网络 (Deep Interest Evolution Network, DIEN)**
  - 实现文件：`dien.py`
  - DIN的升级版，增加了兴趣演化建模
  - 通过兴趣抽取层和兴趣演化层捕获用户兴趣的动态变化
  - 利用辅助网络提供额外监督信号
  - 采用GRU捕获序列中的时间依赖关系

- **基于Transformer的推荐 (Transformer-based Recommendation)**
  - 实现文件：`transformer_rec.py`
  - 使用自注意力机制捕捉用户行为序列中的长期和短期兴趣
  - 实现了SASRec(Self-Attentive Sequential Recommendation)的核心思想
  - 通过位置编码处理序列顺序信息

### 3. 图神经网络推荐算法

- **LightGCN**
  - 实现文件：`gnn_rec.py`
  - 简化的图卷积网络推荐算法
  - 通过用户-物品交互构建二部图，利用图卷积进行消息传递
  - 移除了传统GCN中的特征变换和非线性激活，提高性能

### 4. 多任务学习推荐算法

- **MMoE (Multi-gate Mixture-of-Experts)**
  - 实现文件：`multitask_rec.py`
  - 多任务学习框架，可以同时优化多个目标(如点击率、转化率)
  - 使用门控机制动态调整各专家网络对不同任务的贡献
  - 有效处理任务之间的相关性和冲突

### 5. 强化学习推荐算法

- **DQN推荐器 (Deep Q-Network Recommender)**
  - 实现文件：`rl_rec.py`
  - 使用深度Q网络进行序列决策推荐
  - 将推荐过程建模为马尔可夫决策过程(MDP)
  - 通过经验回放和目标网络提高训练稳定性
  - 适用于考虑长期用户满意度的推荐场景

### 6. 大语言模型推荐算法

- **LLM推荐器 (Large Language Model Recommender)**
  - 实现文件：`llm_rec.py`
  - 利用大语言模型理解用户兴趣和物品内容
  - 支持推荐解释和个性化描述生成
  - 包含混合大模型推荐算法(LLMHybridRecommender)，结合传统推荐和LLM能力
  - 可用于提供内容丰富、解释性强的推荐结果

## 使用方法

所有算法均遵循相似的接口设计，通常包括以下方法：

- `__init__`: 初始化模型和参数
- `fit`/`train_model`: 训练模型
- `predict`: 预测用户对物品的评分/喜好
- `recommend`: 为用户生成推荐列表
- `save_model`/`load_model`: 保存和加载模型
- `explain`: 部分算法支持推荐解释（如LLM推荐器）

示例代码：

```python
# 使用协同过滤
from collaborative_filtering import CollaborativeFiltering

# 初始化模型
cf = CollaborativeFiltering(method='user')

# 训练模型
cf.fit(ratings_matrix)

# 为用户0推荐物品
recommendations = cf.recommend(ratings_matrix, 0)
```

对于深度学习模型，一般使用方式如下：

```python
# 使用DeepFM
from deepfm import DeepFM

# 初始化模型
model = DeepFM(feature_size=feature_size, field_size=field_size)

# 训练模型
train_data = (feature_idx, feature_values, labels)
model.train_model(train_data, epochs=10)

# 预测
predictions = model.predict(test_features, test_values)
```

更多使用示例可以查看`main.py`文件。

## 环境依赖

项目依赖以下库：

- PyTorch >= 1.10.0
- NumPy >= 1.19.5
- SciPy >= 1.7.0
- Scikit-learn >= 1.0.0
- Pandas >= 1.3.0（用于数据处理）
- Matplotlib >= 3.4.0（用于可视化，可选）

可以通过以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 运行示例

通过运行`main.py`可以测试所有已实现的算法：

```bash
python main.py
```

这将生成示例数据并对每种算法进行测试。你可以在控制台看到每个算法的输出结果和性能指标。

## 算法比较

不同算法适用于不同场景：

- **协同过滤/矩阵分解**: 适用于简单场景，数据稀疏时表现较差，计算效率高
- **基于内容的推荐**: 适用于冷启动问题，不受用户交互数据限制，但需要高质量的内容特征
- **深度学习模型**:
  - **NCF**: 较好地平衡了精度和复杂度，适合中小规模系统
  - **Wide&Deep/DeepFM/xDeepFM**: 适用于大规模数据集和复杂特征建模，特别是有大量类别特征和数值特征的场景
  - **DIN/DIEN**: 适用于电商和广告推荐，能有效捕捉用户兴趣变化
  - **Transformer**: 适用于序列推荐，能有效处理长距离依赖问题
- **图神经网络(LightGCN)**: 利用高阶连接信息，适合社交网络相关推荐，能缓解数据稀疏问题
- **多任务学习(MMoE)**: 适用于需要同时优化多个目标的场景，如同时提高点击率和转化率
- **强化学习推荐**: 适用于长期用户满意度优化和多步骤交互场景，能够考虑推荐的长期影响
- **大语言模型推荐**: 适用于内容丰富、需要解释性的推荐场景，能提供个性化的推荐理由

## 模型性能对比

在一般的电子商务推荐场景中，各算法的相对性能（以AUC为指标）大致如下：

1. xDeepFM > DeepFM > Wide&Deep > NCF
2. DIEN > DIN > Transformer > 传统方法
3. 大语言模型推荐在内容丰富场景下表现最佳

## 扩展与自定义

系统设计支持灵活扩展：
- 新算法可以通过实现相同接口被轻松集成
- 现有算法可以通过继承和重写来自定义
- 模型组件（如注意力层、嵌入层等）可以单独使用

## 参考论文

- DeepFM: [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)
- xDeepFM: [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/abs/1803.05170)
- DIN: [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)
- DIEN: [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/abs/1809.03672)
- LightGCN: [LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation](https://arxiv.org/abs/2002.02126)
- SASRec: [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- MMoE: [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/10.1145/3219819.3220007)
- Wide&Deep: [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- NCF: [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)
- DQN: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) 