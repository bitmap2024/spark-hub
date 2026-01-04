import numpy as np
import torch
import scipy.sparse as sp
from collaborative_filtering import CollaborativeFiltering
from content_based import ContentBasedRecommender
from matrix_factorization import MatrixFactorization
from neural_cf import NeuralCollaborativeFiltering
from hybrid import HybridRecommender
from transformer_rec import TransformerRecommender
from gnn_rec import LightGCN
from multitask_rec import MMoE
from rl_rec import DQNRecommender, RecEnv
from llm_rec import LLMRecommender, LLMHybridRecommender
from wide_deep import WideAndDeep
from deepfm import DeepFM
from xdeepfm import xDeepFM
from din import DIN
from dien import DIEN

def generate_sample_data(n_users=1000, n_items=2000, sparsity=0.05, seq_len=20):
    """
    生成示例数据
    
    参数:
        n_users (int): 用户数量
        n_items (int): 物品数量
        sparsity (float): 稀疏度，表示非零元素的比例
        seq_len (int): 序列长度（用于序列推荐）
        
    返回:
        tuple: (ratings_matrix, items_features, user_sequences)
    """
    # 生成评分矩阵
    n_ratings = int(n_users * n_items * sparsity)
    user_indices = np.random.randint(0, n_users, n_ratings)
    item_indices = np.random.randint(0, n_items, n_ratings)
    ratings = np.random.uniform(1, 5, n_ratings)
    
    ratings_matrix = np.zeros((n_users, n_items))
    ratings_matrix[user_indices, item_indices] = ratings
    
    # 生成物品特征
    items_features = np.random.rand(n_items, 100)  # 100个特征
    
    # 生成用户行为序列
    user_sequences = np.zeros((n_users, seq_len), dtype=np.int64)
    for i in range(n_users):
        seq_length = np.random.randint(5, seq_len+1)
        user_sequences[i, :seq_length] = np.random.choice(n_items, seq_length, replace=False)
    
    # 生成物品文本特征（用于大模型推荐）
    items_text_features = {}
    categories = ["电影", "音乐", "书籍", "游戏", "电子产品", "服装", "食品", "旅游"]
    for i in range(n_items):
        category = np.random.choice(categories)
        title = f"物品_{i}"
        description = f"这是{category}类别的{i}号物品，具有独特的特性。"
        items_text_features[i] = {
            "title": title,
            "category": category,
            "description": description
        }
    
    # 生成用户历史行为（用于大模型推荐）
    user_history = {}
    for i in range(n_users):
        # 每个用户随机选择5-15个物品作为历史
        n_history = np.random.randint(5, 16)
        history_items = np.random.choice(n_items, n_history, replace=False)
        user_history[i] = [{"item_id": item_id} for item_id in history_items]
    
    return ratings_matrix, items_features, user_sequences, items_text_features, user_history

def test_collaborative_filtering(ratings_matrix):
    """测试协同过滤算法"""
    print("\n测试协同过滤算法...")
    cf = CollaborativeFiltering(method='user')
    cf.fit(ratings_matrix)
    prediction = cf.predict(ratings_matrix, 0, 0)
    recommendations = cf.recommend(ratings_matrix, 0)
    print(f"预测评分: {prediction:.4f}")
    print(f"推荐物品: {recommendations}")

def test_content_based(items_features):
    """测试基于内容的推荐算法"""
    print("\n测试基于内容的推荐算法...")
    content_rec = ContentBasedRecommender()
    content_rec.fit(items_features)
    recommendations = content_rec.recommend(0)
    print(f"推荐物品: {recommendations}")

def test_matrix_factorization(ratings_matrix):
    """测试矩阵分解算法"""
    print("\n测试矩阵分解算法...")
    mf = MatrixFactorization(n_factors=50)
    mf.fit(ratings_matrix)
    prediction = mf.predict(0, 0)
    recommendations = mf.recommend(0)
    print(f"预测评分: {prediction:.4f}")
    print(f"推荐物品: {recommendations}")

def test_neural_cf(ratings_matrix):
    """测试神经协同过滤算法"""
    print("\n测试神经协同过滤算法...")
    num_users, num_items = ratings_matrix.shape
    ncf = NeuralCollaborativeFiltering(num_users, num_items)
    
    # 准备训练数据
    user_indices, item_indices = np.where(ratings_matrix > 0)
    ratings = ratings_matrix[user_indices, item_indices]
    
    # 训练模型
    ncf.train_model((user_indices, item_indices, ratings), epochs=5)
    
    # 测试预测
    prediction = ncf.predict(torch.LongTensor([0]), torch.LongTensor([0]))
    recommendations = ncf.recommend(0)
    print(f"预测评分: {prediction[0][0]:.4f}")
    print(f"推荐物品: {recommendations}")

def test_transformer_rec(user_sequences):
    """测试Transformer推荐算法"""
    print("\n测试Transformer推荐算法...")
    num_items = user_sequences.max() + 1
    model = TransformerRecommender(num_items=num_items)
    
    # 准备示例数据
    batch_size = 32
    seq_batch = torch.LongTensor(user_sequences[:batch_size])
    
    # 预测
    predictions = model(seq_batch)
    recommendations = model.predict(seq_batch[0:1])
    print(f"推荐物品: {recommendations}")

def test_lightgcn(ratings_matrix):
    """测试LightGCN算法"""
    print("\n测试LightGCN算法...")
    num_users, num_items = ratings_matrix.shape
    model = LightGCN(num_users=num_users, num_items=num_items)
    
    # 创建邻接矩阵
    adj_matrix = model.create_adj_matrix(sp.csr_matrix(ratings_matrix))
    
    # 预测
    recommendations = model.recommend(0, adj_matrix)
    print(f"推荐物品: {recommendations}")

def test_mmoe(ratings_matrix, items_features):
    """测试MMoE算法"""
    print("\n测试MMoE算法...")
    input_dim = items_features.shape[1]
    model = MMoE(input_dim=input_dim)
    
    # 准备示例数据
    batch_size = 32
    features = torch.FloatTensor(items_features[:batch_size])
    task1_target = torch.FloatTensor(np.random.randint(0, 2, (batch_size, 1)))
    task2_target = torch.FloatTensor(np.random.randint(0, 2, (batch_size, 1)))
    
    # 预测
    predictions = model(features)
    print(f"任务1预测: {predictions[0][0].item():.4f}")
    print(f"任务2预测: {predictions[1][0].item():.4f}")

def test_rl_rec(ratings_matrix, items_features):
    """测试强化学习推荐算法"""
    print("\n测试强化学习推荐算法...")
    
    # 生成用户特征（这里简单使用随机特征）
    user_features = np.random.rand(ratings_matrix.shape[0], 50)
    
    # 创建环境
    env = RecEnv(user_features, items_features, ratings_matrix)
    
    # 创建DQN推荐模型
    state_dim = user_features.shape[1] + items_features.shape[1] + ratings_matrix.shape[1]
    action_dim = ratings_matrix.shape[1]
    model = DQNRecommender(state_dim=state_dim, action_dim=action_dim)
    
    # 训练模型（这里只训练少量回合作为示例）
    model.train_model(env, num_episodes=5)
    
    # 测试预测
    state = env.reset()
    state = torch.FloatTensor(state)
    recommendations = model.predict(state, k=5)
    print(f"推荐物品: {recommendations}")

def test_llm_rec(items_text_features, user_history):
    """测试大模型推荐算法"""
    print("\n测试大模型推荐算法...")
    
    # 创建大模型推荐器
    llm_rec = LLMRecommender()
    
    # 训练模型
    llm_rec.fit(user_history, items_text_features)
    
    # 测试推荐
    user_id = 0
    recommendations = llm_rec.recommend(user_id, user_history[user_id], items_text_features, k=5)
    print(f"推荐物品: {recommendations}")
    
    # 测试推荐解释
    if recommendations:
        item_id = recommendations[0]
        explanation = llm_rec.explain_recommendation(user_id, item_id, user_history[user_id], items_text_features)
        print(f"推荐解释: {explanation}")
        
        # 测试个性化描述
        description = llm_rec.generate_personalized_description(user_id, item_id, user_history[user_id], items_text_features)
        print(f"个性化描述: {description}")

def test_llm_hybrid(ratings_matrix, items_text_features, user_history):
    """测试混合大模型推荐算法"""
    print("\n测试混合大模型推荐算法...")
    
    # 创建混合推荐器
    hybrid_rec = LLMHybridRecommender(llm_weight=0.7)
    
    # 设置传统推荐器（使用协同过滤）
    cf = CollaborativeFiltering(method='user')
    hybrid_rec.set_traditional_recommender(cf)
    
    # 训练模型
    hybrid_rec.fit(user_history, items_text_features, ratings_matrix)
    
    # 测试推荐
    user_id = 0
    recommendations = hybrid_rec.recommend(user_id, user_history[user_id], items_text_features, k=5)
    print(f"混合推荐物品: {recommendations}")

def test_hybrid(ratings_matrix, items_features):
    """测试混合推荐系统"""
    print("\n测试混合推荐系统...")
    hybrid = HybridRecommender()
    hybrid.fit(ratings_matrix, items_features)
    prediction = hybrid.predict(0, 0, ratings_matrix)
    recommendations = hybrid.recommend(0, ratings_matrix)
    print(f"预测评分: {prediction:.4f}")
    print(f"推荐物品: {recommendations}")

def test_wide_deep(ratings_matrix, items_features):
    """测试Wide&Deep算法"""
    print("\n测试Wide&Deep算法...")
    num_users, num_items = ratings_matrix.shape
    num_features = items_features.shape[1]
    
    # 创建Wide&Deep模型
    model = WideAndDeep(
        num_users=num_users,
        num_items=num_items,
        num_features=num_features,
        embedding_dim=32,
        hidden_layers=[64, 32],
        dropout_rate=0.2
    )
    
    # 准备训练数据
    user_indices, item_indices = np.where(ratings_matrix > 0)
    ratings = ratings_matrix[user_indices, item_indices]
    
    # 为每个交互准备特征（这里用物品特征作为示例）
    features = items_features[item_indices]
    
    # 训练模型
    train_data = (user_indices, item_indices, (ratings > 0).astype(float), features)
    model.train_model(train_data, epochs=5, batch_size=256)
    
    # 测试预测
    test_user = torch.LongTensor([0])
    test_item = torch.LongTensor([0])
    test_feature = torch.FloatTensor(items_features[0:1])
    
    prediction = model.predict(test_user, test_item, test_feature)
    
    # 为用户生成推荐
    recommendations = model.recommend(0, n_recommendations=5, features=items_features)
    
    print(f"预测评分: {prediction[0]:.4f}")
    print(f"推荐物品: {recommendations}")

def generate_sequence_samples(n_users=500, n_items=1000, max_hist_len=20, n_samples=5000):
    """
    生成序列推荐的示例数据
    
    参数:
        n_users (int): 用户数量
        n_items (int): 物品数量
        max_hist_len (int): 最大历史长度
        n_samples (int): 样本数量
        
    返回:
        tuple: (user_indices, item_indices, history_indices, history_length, labels)
    """
    user_indices = np.random.randint(0, n_users, n_samples)
    item_indices = np.random.randint(0, n_items, n_samples)
    labels = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    # 为每个用户生成历史行为序列
    history_indices = np.zeros((n_samples, max_hist_len), dtype=np.int64)
    history_length = np.random.randint(1, max_hist_len + 1, n_samples)
    
    for i in range(n_samples):
        seq_len = history_length[i]
        history_indices[i, :seq_len] = np.random.choice(n_items, seq_len, replace=False)
    
    # 为DIEN生成额外数据
    click_indices = np.zeros((n_samples, max_hist_len), dtype=np.int64)
    noclick_indices = np.zeros((n_samples, max_hist_len), dtype=np.int64)
    
    for i in range(n_samples):
        seq_len = history_length[i]
        click_indices[i, :seq_len] = np.random.choice(n_items, seq_len, replace=False)
        noclick_indices[i, :seq_len] = np.random.choice(n_items, seq_len, replace=False)
    
    return user_indices, item_indices, history_indices, history_length, click_indices, noclick_indices, labels

def generate_feature_samples(n_users=500, n_items=1000, n_fields=10, n_samples=5000):
    """
    生成特征工程的示例数据
    
    参数:
        n_users (int): 用户数量
        n_items (int): 物品数量
        n_fields (int): 特征域数量
        n_samples (int): 样本数量
        
    返回:
        tuple: (feature_idx, feature_values, labels)
    """
    # 特征索引，每个样本有n_fields个特征域
    feature_idx = np.random.randint(0, n_users + n_items, (n_samples, n_fields))
    # 特征值（这里简化为随机值）
    feature_values = np.random.random((n_samples, n_fields)).astype(np.float32)
    # 标签
    labels = np.random.randint(0, 2, n_samples).astype(np.float32)
    
    return feature_idx, feature_values, labels

def test_deepfm(feature_idx, feature_values, labels):
    """测试DeepFM算法"""
    print("\n测试DeepFM算法...")
    
    # 获取数据维度
    feature_size = feature_idx.max() + 1
    field_size = feature_idx.shape[1]
    
    # 创建DeepFM模型
    model = DeepFM(
        feature_size=feature_size,
        field_size=field_size,
        embedding_dim=16,
        hidden_layers=[64, 32, 16],
        dropout_rate=0.2
    )
    
    # 训练模型
    train_data = (feature_idx, feature_values, labels)
    model.train_model(train_data, epochs=5, batch_size=256)
    
    # 测试预测
    test_features = torch.LongTensor(feature_idx[:10])
    test_values = torch.FloatTensor(feature_values[:10])
    
    predictions = model.predict(test_features, test_values)
    
    print(f"预测CTR: {predictions[0]:.4f}")
    print(f"前10个预测结果: {[f'{p:.4f}' for p in predictions]}")

def test_xdeepfm(feature_idx, feature_values, labels):
    """测试xDeepFM算法"""
    print("\n测试xDeepFM算法...")
    
    # 获取数据维度
    feature_size = feature_idx.max() + 1
    field_size = feature_idx.shape[1]
    
    # 创建xDeepFM模型
    model = xDeepFM(
        feature_size=feature_size,
        field_size=field_size,
        embedding_dim=16,
        cin_layer_sizes=[64, 32],
        mlp_hidden_layers=[64, 32],
        dropout_rate=0.2
    )
    
    # 训练模型
    train_data = (feature_idx, feature_values, labels)
    model.train_model(train_data, epochs=5, batch_size=256)
    
    # 测试预测
    test_features = torch.LongTensor(feature_idx[:10])
    test_values = torch.FloatTensor(feature_values[:10])
    
    predictions = model.predict(test_features, test_values)
    
    print(f"预测CTR: {predictions[0]:.4f}")
    print(f"前10个预测结果: {[f'{p:.4f}' for p in predictions]}")

def test_din(user_indices, item_indices, history_indices, history_length, labels):
    """测试DIN算法"""
    print("\n测试DIN算法...")
    
    # 获取数据维度
    num_users = user_indices.max() + 1
    num_items = max(item_indices.max(), history_indices.max()) + 1
    
    # 创建DIN模型
    model = DIN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=32,
        mlp_layers=[64, 32],
        dropout_rate=0.2,
        max_hist_len=history_indices.shape[1]
    )
    
    # 训练模型
    train_data = (user_indices, item_indices, history_indices, history_length, labels)
    model.train_model(train_data, epochs=5, batch_size=256)
    
    # 测试预测
    test_users = torch.LongTensor(user_indices[:10])
    test_items = torch.LongTensor(item_indices[:10])
    test_history = torch.LongTensor(history_indices[:10])
    test_length = torch.LongTensor(history_length[:10])
    
    predictions = model.predict(test_users, test_items, test_history, test_length)
    
    print(f"预测CTR: {predictions[0]:.4f}")
    print(f"前10个预测结果: {[f'{p:.4f}' for p in predictions]}")
    
    # 测试推荐
    user_id = 0
    history = torch.LongTensor(history_indices[0])
    length = history_length[0]
    
    recommendations = model.recommend(user_id, history, length, n_recommendations=5)
    print(f"为用户{user_id}推荐的物品: {recommendations}")

def test_dien(user_indices, item_indices, history_indices, history_length, click_indices, noclick_indices, labels):
    """测试DIEN算法"""
    print("\n测试DIEN算法...")
    
    # 获取数据维度
    num_users = user_indices.max() + 1
    num_items = max(item_indices.max(), history_indices.max(), 
                   click_indices.max(), noclick_indices.max()) + 1
    
    # 创建DIEN模型
    model = DIEN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=32,
        hidden_size=32,
        mlp_layers=[64, 32],
        dropout_rate=0.2,
        max_hist_len=history_indices.shape[1]
    )
    
    # 训练模型
    train_data = (user_indices, item_indices, history_indices, history_length, 
                 click_indices, noclick_indices, labels)
    model.train_model(train_data, epochs=5, batch_size=256)
    
    # 测试预测
    test_users = torch.LongTensor(user_indices[:10])
    test_items = torch.LongTensor(item_indices[:10])
    test_history = torch.LongTensor(history_indices[:10])
    test_length = torch.LongTensor(history_length[:10])
    
    predictions = model.predict(test_users, test_items, test_history, test_length)
    
    print(f"预测CTR: {predictions[0]:.4f}")
    print(f"前10个预测结果: {[f'{p:.4f}' for p in predictions]}")
    
    # 测试推荐
    user_id = 0
    history = torch.LongTensor(history_indices[0])
    length = history_length[0]
    
    recommendations = model.recommend(user_id, history, length, n_recommendations=5)
    print(f"为用户{user_id}推荐的物品: {recommendations}")

def main():
    """主函数"""
    print("生成示例数据...")
    ratings_matrix, items_features, user_sequences, items_text_features, user_history = generate_sample_data()
    
    # 生成序列推荐和特征工程的示例数据
    print("生成序列推荐和特征工程的示例数据...")
    user_indices, item_indices, history_indices, history_length, click_indices, noclick_indices, labels = generate_sequence_samples()
    feature_idx, feature_values, feature_labels = generate_feature_samples()
    
    # 测试各个推荐算法
    test_collaborative_filtering(ratings_matrix)
    test_content_based(items_features)
    test_matrix_factorization(ratings_matrix)
    test_neural_cf(ratings_matrix)
    test_transformer_rec(user_sequences)
    test_lightgcn(ratings_matrix)
    test_mmoe(ratings_matrix, items_features)
    test_rl_rec(ratings_matrix, items_features)
    test_wide_deep(ratings_matrix, items_features)
    test_llm_rec(items_text_features, user_history)
    test_llm_hybrid(ratings_matrix, items_text_features, user_history)
    test_hybrid(ratings_matrix, items_features)
    
    # 测试新添加的算法
    test_deepfm(feature_idx, feature_values, feature_labels)
    test_xdeepfm(feature_idx, feature_values, feature_labels)
    test_din(user_indices, item_indices, history_indices, history_length, labels)
    test_dien(user_indices, item_indices, history_indices, history_length, click_indices, noclick_indices, labels)

if __name__ == "__main__":
    main()
