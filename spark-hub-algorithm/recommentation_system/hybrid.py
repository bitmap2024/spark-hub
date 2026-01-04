import numpy as np
from .collaborative_filtering import CollaborativeFiltering
from .content_based import ContentBasedRecommender
from .matrix_factorization import MatrixFactorization
from .neural_cf import NeuralCollaborativeFiltering

class HybridRecommender:
    """
    混合推荐系统
    
    结合多种推荐算法的优势，提供更准确的推荐
    """
    def __init__(self, weights=None):
        """
        初始化混合推荐系统
        
        参数:
            weights (dict): 各个推荐器的权重，默认为None，表示使用平均权重
        """
        self.cf_model = CollaborativeFiltering()
        self.content_model = ContentBasedRecommender()
        self.mf_model = MatrixFactorization()
        self.ncf_model = None  # 延迟初始化，因为需要知道用户和物品数量
        
        # 设置默认权重
        self.weights = weights or {
            'cf': 0.3,
            'content': 0.2,
            'mf': 0.3,
            'ncf': 0.2
        }
        
    def fit(self, ratings_matrix, items_features, train_ncf=True):
        """
        训练混合推荐系统
        
        参数:
            ratings_matrix (numpy.ndarray): 用户-物品评分矩阵
            items_features (numpy.ndarray): 物品特征矩阵
            train_ncf (bool): 是否训练神经协同过滤模型
            
        返回:
            self: 返回模型自身，支持链式调用
        """
        # 训练协同过滤模型
        self.cf_model.fit(ratings_matrix)
        
        # 训练基于内容的模型
        self.content_model.fit(items_features)
        
        # 训练矩阵分解模型
        self.mf_model.fit(ratings_matrix)
        
        # 训练神经协同过滤模型
        if train_ncf:
            num_users, num_items = ratings_matrix.shape
            self.ncf_model = NeuralCollaborativeFiltering(num_users, num_items)
            
            # 准备训练数据
            user_indices, item_indices = np.where(ratings_matrix > 0)
            ratings = ratings_matrix[user_indices, item_indices]
            
            # 训练模型
            self.ncf_model.train_model((user_indices, item_indices, ratings))
        
        return self
    
    def predict(self, user_id, item_id, ratings_matrix):
        """
        预测用户对物品的评分
        
        参数:
            user_id (int): 用户ID
            item_id (int): 物品ID
            ratings_matrix (numpy.ndarray): 用户-物品评分矩阵
            
        返回:
            float: 预测的评分
        """
        predictions = {}
        
        # 获取各个模型的预测
        predictions['cf'] = self.cf_model.predict(ratings_matrix, user_id, item_id)
        predictions['mf'] = self.mf_model.predict(user_id, item_id)
        
        if self.ncf_model is not None:
            import torch
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            predictions['ncf'] = self.ncf_model.predict(user_tensor, item_tensor)[0][0]
        
        # 计算加权平均
        final_prediction = 0
        total_weight = 0
        
        for model_name, prediction in predictions.items():
            if model_name in self.weights:
                final_prediction += prediction * self.weights[model_name]
                total_weight += self.weights[model_name]
        
        return final_prediction / total_weight if total_weight > 0 else 0
    
    def recommend(self, user_id, ratings_matrix, n_recommendations=5):
        """
        为用户推荐物品
        
        参数:
            user_id (int): 用户ID
            ratings_matrix (numpy.ndarray): 用户-物品评分矩阵
            n_recommendations (int): 推荐物品数量
            
        返回:
            list: 推荐的物品ID列表
        """
        num_items = ratings_matrix.shape[1]
        predictions = np.zeros(num_items)
        
        # 获取用户未评分的物品
        unrated_items = np.where(ratings_matrix[user_id] == 0)[0]
        
        # 为每个未评分的物品计算预测评分
        for item_id in unrated_items:
            predictions[item_id] = self.predict(user_id, item_id, ratings_matrix)
        
        # 返回评分最高的物品
        recommendations = np.argsort(predictions)[-n_recommendations:]
        return recommendations.tolist()
    
    def update_weights(self, new_weights):
        """
        更新各个推荐器的权重
        
        参数:
            new_weights (dict): 新的权重字典
        """
        self.weights.update(new_weights)
        
        # 归一化权重
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total 