import numpy as np
import matplotlib.pyplot as plt
from wide_deep import WideAndDeep

# 创建一些模拟数据
def generate_synthetic_data(n_users=1000, n_items=500, n_features=10, sparsity=0.95):
    """
    生成合成数据用于测试Wide&Deep模型
    
    参数:
        n_users (int): 用户数量
        n_items (int): 物品数量
        n_features (int): 特征数量
        sparsity (float): 稀疏程度 (0-1)
    
    返回:
        tuple: (user_indices, item_indices, ratings, features)
    """
    # 创建用户-物品交互矩阵 (稀疏)
    n_interactions = int(n_users * n_items * (1 - sparsity))
    print(f"生成 {n_interactions} 个交互数据...")
    
    user_indices = np.random.randint(0, n_users, n_interactions)
    item_indices = np.random.randint(0, n_items, n_interactions)
    
    # 创建特征矩阵 (每个物品有n_features个特征)
    features = np.random.randn(n_interactions, n_features)
    
    # 创建评分数据 (二元评分: 0或1)
    # 使用一些特征和用户-物品交互来生成评分
    ratings = np.zeros(n_interactions)
    for i in range(n_interactions):
        user = user_indices[i]
        item = item_indices[i]
        
        # 一些随机规则决定评分
        if (user % 3 == 0 and item % 5 == 0) or np.sum(features[i]) > 0:
            ratings[i] = 1.0
        elif user % 7 == 0 or (item % 2 == 0 and np.mean(features[i]) > 0):
            ratings[i] = 1.0
    
    print(f"正面评分比例: {np.mean(ratings):.2f}")
    return user_indices, item_indices, ratings, features

def main():
    print("Wide&Deep推荐系统示例")
    print("------------------------")
    
    # 生成数据
    n_users = 1000
    n_items = 500
    n_features = 20
    user_indices, item_indices, ratings, features = generate_synthetic_data(
        n_users, n_items, n_features
    )
    
    # 划分训练集和测试集
    n_samples = len(user_indices)
    indices = np.random.permutation(n_samples)
    train_size = int(n_samples * 0.8)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 训练数据
    train_users = user_indices[train_indices]
    train_items = item_indices[train_indices]
    train_ratings = ratings[train_indices]
    train_features = features[train_indices]
    
    # 测试数据
    test_users = user_indices[test_indices]
    test_items = item_indices[test_indices]
    test_ratings = ratings[test_indices]
    test_features = features[test_indices]
    
    # 创建Wide&Deep模型
    print("\n创建Wide&Deep模型...")
    model = WideAndDeep(
        num_users=n_users,
        num_items=n_items,
        num_features=n_features,
        embedding_dim=32,
        hidden_layers=[64, 32, 16],
        dropout_rate=0.2
    )
    
    # 训练模型
    print("\n开始训练模型...")
    train_data = (train_users, train_items, train_ratings, train_features)
    history = model.train_model(
        train_data, 
        epochs=5, 
        batch_size=256, 
        learning_rate=0.001
    )
    
    # 评估模型
    print("\n评估模型...")
    import torch
    model.eval()
    with torch.no_grad():
        test_user_tensor = torch.LongTensor(test_users)
        test_item_tensor = torch.LongTensor(test_items)
        test_feature_tensor = torch.FloatTensor(test_features)
        
        predictions = model.forward(test_user_tensor, test_item_tensor, test_feature_tensor)
        predictions = predictions.cpu().numpy()
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # 计算AUC
    auc = roc_auc_score(test_ratings, predictions)
    print(f"测试集AUC: {auc:.4f}")
    
    # 计算准确率
    predicted_labels = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(test_ratings, predicted_labels)
    print(f"测试集准确率: {accuracy:.4f}")
    
    # 可视化损失历史
    plt.figure(figsize=(10, 5))
    plt.plot(history)
    plt.title('Wide&Deep模型训练损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失')
    plt.grid(True)
    plt.savefig('wide_deep_training_loss.png')
    print("训练损失曲线已保存到 'wide_deep_training_loss.png'")
    
    # 为随机用户生成推荐
    print("\n为随机用户生成推荐...")
    random_user = np.random.randint(0, n_users)
    
    # 创建物品特征矩阵 (随机生成)
    all_item_features = np.random.randn(n_items, n_features)
    
    # 获取推荐
    recommendations = model.recommend(random_user, n_recommendations=10, features=all_item_features)
    print(f"用户 {random_user} 的推荐物品: {recommendations}")
    
    # 保存模型
    print("\n保存模型...")
    model.save_model("wide_deep_model.pth")
    print("模型已保存到 'wide_deep_model.pth'")

if __name__ == "__main__":
    main() 